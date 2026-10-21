"""Call-scoped Ragas transport observations; no credentials or shared last-result state."""
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import version
import json
import asyncio
import time
from urllib.parse import urlsplit,urlunsplit

_ACTIVE=ContextVar('ragas_evaluation_observation',default=None)


class Recorder:
    def __init__(self,limit,timeout):
        if type(limit) is not int or not 1<=limit<=32 or type(timeout) not in (int,float) or not 0<timeout<=600:
            raise ValueError('RAGAS_OBSERVATION_BUDGET_INVALID')
        self.limit=limit;self.timeout=timeout;self.rows=[];self.inflight={};self.metric=None;self.total_timeout=300
    async def request(self,request):
        if len(self.rows)>=self.limit:raise RuntimeError('RAGAS_EVALUATION_CALL_BUDGET_EXHAUSTED')
        parsed=urlsplit(str(request.url))
        url=urlunsplit((parsed.scheme,parsed.hostname+(':'+str(parsed.port) if parsed.port else ''),parsed.path,'',''))
        body=json.loads(request.content)
        row={'metric':self.metric,'kind':'embedding' if parsed.path.endswith('/embeddings') else 'llm',
             'url':url,'request':body,'started_monotonic':time.monotonic(),'status':'request_started'}
        self.rows.append(row);self.inflight[id(request)]=row
    async def response(self,response):
        row=self.inflight[id(response.request)]
        row.update(http_status=response.status_code,elapsed_seconds=time.monotonic()-row['started_monotonic'])
        raw=await response.aread()
        try:body=json.loads(raw)
        except ValueError:
            row['status']='invalid_response_json';return
        row.update(status='response_received',response=body,usage=body.get('usage'))
    def cost(self):
        llm=[r for r in self.rows if r['kind']=='llm']
        result={'model_calls':len(llm),'embedding_calls':len(self.rows)-len(llm)}
        for output,source in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
            values=[(r.get('usage') or {}).get(source) for r in llm]
            result[output]=sum(values) if all(type(v) is int and v>=0 for v in values) else None
        values=[((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens') for r in llm]
        result['cached_input_tokens']=sum(values) if all(type(v) is int and v>=0 for v in values) else None
        return result


@contextmanager
def capture(sink,settings,metrics,*,limit=8,timeout=180):
    if sink is None:
        yield;return
    recorder=Recorder(limit,timeout);token=_ACTIVE.set(recorder);start=time.monotonic()
    try:yield
    finally:
        _ACTIVE.reset(token)
        cfg=settings.llm if settings is not None else None
        sink({'cost':recorder.cost(),'requests':recorder.rows,'elapsed_seconds':time.monotonic()-start,
              'evaluator_identity':{'ragas_version':version('ragas'),'model':getattr(cfg,'model',None),
                  'provider':getattr(cfg,'provider',None),'temperature':getattr(cfg,'temperature',None),'max_tokens':getattr(cfg,'max_tokens',None),'metrics':list(metrics)},
              'scope':'Actual HTTP request attempts; failed requests keep unknown usage; no headers or settings snapshot.'})


def client_options():
    current=_ACTIVE.get()
    if current is None:return {}
    import httpx
    # Scope: HTTP observation only. 为什么要有它：Ragas一项分数可能包含多次模型调用，必须记录真实请求，不能将一次evaluate当成一次模型调用。
    return {'http_client':httpx.AsyncClient(timeout=current.timeout,event_hooks={'request':[current.request],'response':[current.response]})}

def metric_started(name):
    current=_ACTIVE.get()
    if current is not None:current.metric=name


async def within_budget(coroutine):
    current=_ACTIVE.get()
    return await asyncio.wait_for(coroutine,current.total_timeout) if current is not None else await coroutine
