# Project State

> **历史快照（主线整合基线 `0e688f9`）。** 下文保留原有状态、日期和兼容锚点；“当前”描述的是记录时点。现行工作归属请从[文档总入口](README.md#当前工作线)、[统一登记表](document-registry.json)和 [GitHub 任务](https://github.com/yang-hoo-xin/RepoForge/issues)查找。
>
> 为什么要有它：多个分支保留了不同状态，旧快照不应被新的聊天当作自动续作指令。

更新于 2026-10-09。范围：当前工作区可定位的实现、已执行交付和明确缺口；不是通用成功率或发布验收声明。

## 1. 当前状态

**10月9日服务化收口后恢复十题：** C3 Codex 按原 gpt-5.6-sol/medium、600秒完成442.234秒运行，宿主受控验收通过，只有Plugin.cs改变，无越界。165份缺失依赖已按冻结SHA恢复；模型内公开checker因manifest仍指向被拒绝的旧目录而不可用，原请求重连也保留，不能宣称与平台反馈机会完全相同或整题已交付。正修JIT清单重绑定，并将资格从仅--help补至实际原缺陷行为检查。C3平台昨日311.953秒用户中断原件和锁不改，另准备第二次完整新尝试；Ollama实际0.35.1→0.40.1已记录，不混为同版本耗时对照。C4旧验收只有编译、无法拒绝原Hook缺陷，正在新版本补受控行为正负控制，实机仍待验。C2既有Codex候选及日志43份已逐字保留在 `artifacts/eval_driven_20261008/benchmark10/C2/codex-retained-v1/index.json`；只是归档，不是补跑或行为通过。

**10月9日按最新要求优先完成服务化层：** 已交付 FastAPI、独立 Worker、Redis 队列/缓存/限流/幂等、租约/超时/重试、Prometheus、Docker Compose，并实际执行五组 Locust。纯 BM25 小语料在两 Worker、4 并发条件下完成64/64，1.028任务/秒、P95 4.031秒；12并发吞吐0.993、P95 12.437秒，未证明继续增并发或缓存使端到端提速。真实 Agent HTTP 新任务179.085秒，候选编译和15/15断言通过，到达待审批，未Apply，不计十题交付。故障联调覆盖真实Redis中断、Worker死亡与租约接管。原失败保留；专用本机模型/API/Worker已按所有权停止，Docker五个服务保留运行，未触碰游戏。[交付、范围和原始证据入口](../artifacts/eval_driven_20261009/service/RESULTS.md)、[部署文档](../deployment/service/README.md)。十题和独立优化对照仍保留10月8日暂停进度，以下按日期保留历史，不能当作今日在途状态。

**10月8日18:32按用户要求暂停：** C3平台在途运行已按所有权停止，原runner终态与部分原件保留，另记user_interrupted，不算模型自然失败；本轮专用Ollama服务及其runner已关闭，C3与GTFO进程均为零。C2 Codex此前已248秒生成并编译候选，门行为仍未验。剩余队列及中断指针见 `artifacts/eval_driven_20261008/benchmark10/current-queue.json` 和 `user-stop-20261008/pause-state.json`；没有派发后续模型、Build或游戏。

**10月8日原十题正在实际执行：** 服务化及独立优化20槽暂停。S3双方37/37，平台345.265秒、Codex207.468秒；S4双方69/69，平台341秒、Codex环境修正后新运行197.281秒，旧217.547秒访问故障未覆盖。C1平台600.063秒超时，曾产生候选但编译失败，随后修订未完成；Codex567.828秒，独立验收46/46。以上均为模型＋受控阶段，实机仍pending，不计整题交付、不追改600秒时钟。对应S3/S4持久原件与摘要位于 `artifacts/eval_driven_20261008/benchmark10/{S3,S4}`；当前运行及剩余队列见 `artifacts/eval_driven_20261008/benchmark10/current-queue.json`。五道复杂题工件现已具备模型入口；C4实际编译和生产Hook查找复现MissingMethodException，但不能据旧目录标签断言历史游戏release日期；C5仍缺真实第二客户端。

**Codex执行环境已纠正并真实验证：** 提前创建的未来目录可能继承先前沙箱拒绝ACE，原S1又将执行器根列入deny；S1本次260.812秒访问失败及源未改记录保留。`runner-environment-v3/jit_codex_v3.py`使用同SHA四个CLI文件的专用目录，临运行才复制同源工作区，保持其他题、私有工件和游戏目录拒绝；不手动移除旧ACL。真实native sandbox已验证自有读写、公开读取、私有假哨兵拒绝、checker启动和.NET，C1实际模型运行随后成功。旧准备/权限失败、资格60秒超时及模型前输出目录冲突均独立保留，不当成模型解题失败；此处没有改模型、推理档位或预算。

**S5开发者辅助闭环已完成，原Agent成绩未变：** `artifacts/eval_driven_20261007/S5-closeout/result.json`及同日`S5-closeout-evidence.zip`保存直接修复后的54/54受控检查、正常游戏47/47复核和临时环境恢复。原截图记录字段缺失导致的首个infra_failed及后续同原图复评都保留。这是开发者辅助修复与补验，不能增加产品模型自主交付数。Codex后续独立尝试 `Temp/b10s5c_1007a/results/result.json` 在600.031秒正式验收超时，虽然Agent内部最后一轮54/54通过，仍不是完整交付。

**10月7日热启动对照已执行12/20槽：** S2候选第二次`s2c2`正式DLL35/35通过，698.454秒，预热31.641秒另计，仅HudSettings.cs变化；路由加产品40,186输入/2,466输出Token，预热13/1另计。随后S2基线第二次`s2b2`在879.562秒因`PATCH_BEHAVIOR_OBSERVATION_BUDGET_EXHAUSTED`失败，9次产品调用全部实际返回，未Apply，不是provider或900秒超时。S2两组各有一次成功、一次失败，不能只选双方成功的不同重复组成快配对。原成绩保留，[前11槽汇总](../artifacts/eval_driven_20261006/context-online-ab/summary-hot-first11-v3_7.json)仍是当时状态；第12槽原件在`Temp/coab_hot2_20261006/results/s2b2`，新分层汇总待3.8定向检查后生成。下一原槽为spb2。

**跨日执行环境需分层：** 首8槽实际FA enabled、batch/ubatch512、GPU37/49层；今日S5两槽及S2第二次双方实际FA disabled、256、23/49层。今日服务显式设置`OLLAMA_FLASH_ATTENTION=false`，原服务未设置；Ollama 0.35.1启动日志对两种情况均显示false，但native参数分别off与auto，不能照抄该显示值恢复原语义。模型digest/64K与非零驻留相同不足以证明相同执行条件。[原始日志摘录和逐槽绑定v2](../artifacts/eval_driven_20261006/context-online-ab/actual-host-conditions-v2.json)保留全部原hot_verified和成绩，按实际环境解释；相同环境双方均成功的同题同重复才配对。汇总器3.8代码已准备，尚未测试，不用未验证输出支撑结论。S2结束后已按用户新确认的跨页协调授权，将短CPU窗口交另一开发页；本页暂不启动下一槽或改服务。

**C3可手动运行的初始工程已编译验证：** ExpeditionFocus提供F6手动切换两档完整preset，实际消费在面板与中央准星；首次Build0警告/0错误，真实preset文件与编译控制器18/18通过，显示边界为替身，未运行BasePlugin、Unity或LevelAPI，未验证提交后回滚。自动关卡切换尚未实现，两个实际关卡映射及整题裁判资格仍缺。[初始工程与检查原件](../artifacts/eval_driven_20261007/benchmark10/C3/host/preparation-v2.json)。用户已明确确认“未映射Expedition保持当前有效配置”，写入acceptance-draft-v3；旧暂定草案保留，确认不增加通过数。

**10月7日S5反馈压缩在线v3已结束：** 从同一原始工程重跑，平台快照仅叠加反馈编码与资格身份绑定（575文件）；保留240秒单请求、18次调用和900秒总预算。候选编译通过，54项37过/17失败；第三次Patch实际发送的原生请求含17条完整失败观察，使用实际执行快照还原逐值一致，反馈与源码观察合计23,940字节低于32,768并准入。随后该修订请求240秒无响应，整轮834.797秒失败，6次实际发送/5次完整回复；已知29,832输入/2,100输出Token，末次和完整总成本未知，预热1.25秒13/1另计。[运行、真实压缩请求与成本原件](../artifacts/eval_driven_20261007/S5-feedback-online-v3/prepared.json)。无Apply、正式Build或游戏启动，超时后本地模型已卸载；只能确认反馈预算阻断已被越过，不能声称修复成功或交付提速。准备时复核发现finalizer和batch仍指旧core，已在Agent启动前另存修正版并通过真实composite加载核验；原准备和旧20槽均保留。

**C3身份资料已实际补取：** 现有PEReader读取正常游戏3个DLL并绑定SHA/MVID；确认RundownManager的ActiveExpeditionUniqueKey/ActiveExpedition访问器，以及GTFO.API.ActiveExpedition中的rundownKey/tier/expeditionIndex与独立种子字段。GameData-ASM的相关查询为零命中，ExpeditionInTierData实际位于Modules-ASM，不据零命中写成通过。[保留的原始元数据](../artifacts/eval_driven_20261007/benchmark10/C3/public/materials/current-identity-v1/index.json)。这是只读接口证据；C3初始工程、preset应用事务、裁判资格与两关实机切换尚未完成。

**10月7日最新S5状态：** 用户已确认“今天可以，启动时提醒我”，仅表示配合时间可用，不作为已到菜单或实机证据。RAG CPU候选28答已结束并归还资源。热启动第9、10槽S5均在自然路由处要求澄清：模型选择configured_project却又将同一项目标为reference_only，被原门禁拒绝；没有产品生成或游戏验收。[十次终态汇总](../artifacts/eval_driven_20261006/context-online-ab/summary-hot-first10-v3_7.json)。原20槽下一项仍为s2c2。

独立显式目标S5开发尝试运行572.125秒后失败，第二个Patch请求180.145秒无响应；并非64K上下文准入失败，未生成候选或启动游戏。四个已完成阶段已知21,793输入/1,245输出Token，超时调用用量未知，不能当完整成本。原终态和原件保留在`Temp/s5x_20261007/results`。服务端请求清理后，另开新版本仅将单请求超时改为240秒，仍18次产品调用/900秒完整交付预算；固定1 Token预热26.375秒、13输入/1输出另计。新尝试从原源码重新运行，不是原任务续接，也不属于20槽，[配置和原失败绑定](../artifacts/eval_driven_20261007/S5-explicit-target-v2/selection-origin.json)。当前运行结果以`Temp/s5x2_20261007/results/result.json`终态为准，不能以本段进行中记录判定成功。

**S5开发v2已终态：** 共618.062秒、5次实际返回的产品请求、29,862输入/2,166输出Token（预热另计，缓存尚未在本条对账）。第二个Patch请求205.118秒返回完整候选，随后行为失败反馈累计47,398字节超过32,768，未能进入下一次修订；剩13次调用额度，原源码未Apply，未启动游戏。已有压缩实现对本次原反馈零模型回放后累计24,312字节，原始观察全部可还原，但尚无新模型读取压缩反馈或在线修复结果。[原始终态、成本、回放与验证引用](../artifacts/eval_driven_20261007/S5-explicit-target-v2/prepared.json)。v2增加了独立预热记录，不能将整体运行差异归为timeout单因素收益。工作区另修正误导性的角色冲突澄清问句，拒绝门禁不变；正常入口与澄清续接59项首次通过，不代表模型角色识别准确率提升。

候选实际编译通过，3个受控进程正常退出，54断言37过/17失败：7项渲染未回主线程、9项无关配置丢失、1项原子替换未更新。正式Build计数0不代表候选没有编译。现有续接分别要求尚未发送、已保存但尚未终态的模型输出，或验收环境故障，均不适用于本次终态行为失败；后续应以独立开发版本准入，不能延长原900秒或伪造中断恢复。正常GTFO仍未启动，用户的“今天可以”保留为可配合时间信息。

**10月7日反馈表示复验：** 最新短引用及重复字段列表示40项定向检查通过；原ConfigMerge反馈29562字节准确复现，新投影20929，累计32048低于原32768预算，全部观察可还原。[离线回放v4](../artifacts/eval_driven_20261006/context-online-ab/development-verification/cmc1-feedback-replay-v4.json)。这只解决该失败输入的表示预算，尚无新Agent修订、Token/耗时或交付收益；下方v3未装入预算记录保留为历史经过。

**10月7日中断续接与实际复验：** 已核对原进程全部退出，未重跑已结束模型槽位。源码材料预算定向验证61个不同用例均有通过记录，首轮3失败和复验原件保留；缓存/host接线243项通过（首轮235通过/8临时目录权限失败，8项单独复验通过）。反馈首版57项通过，前缀门槛修订后15项与旧反馈22项均已通过，保留一项要求无净收益样本也编码的测试假设失败及修正。原ConfigMerge反馈字节29562被准确复现，新投影23001，全部原始观察可还原；累计34120仍大于原32768预算，[零模型回放](../artifacts/eval_driven_20261006/context-online-ab/development-verification/cmc1-feedback-replay-v3.json)，不能记作预算阻断已解决或新Agent交付。

C1正控制真实Build0/21检查通过；6个负控制均已有真实Build0后的行为拒绝，5次长路径依赖复制失败另存，10月7日用短run-id补验，不冒充首次成功。C2初始、7控制、观察器和诊断器均编译通过，两个正控制策略诊断各7/7；缺安全门发现的负控制需实机，未以纯策略诊断替代。C1/C2均尚缺相应实机资格和正式受测Agent运行。

RAG页原56答已完成；CPU候选只跑资格且语义失败，主28答尚未开始，原失败保留。今日已在离线窗口结束后明确交回CPU/模型使用权，由该页恢复服务并续未尝试槽位；本页不启动模型/游戏或重测试，待明确归还。为什么要有它：中断后先核对实际进程和原件，新增验证与原冻结模型成绩分开，不把准备失败、部分节省或编译通过当交付。

**统一热启动对照：** 用户要求排除模型加载差异，已将原冷/热探索记录单列，另建同5题×两策略×两次的热启动版本。两组复用同一574文件产品快照和初始工件，正式预算900秒；每次先执行固定不含任务内容的1 Token预热，核对实际模型digest/64K上下文/单模型驻留及非零GPU占用（允许CPU/GPU混合，不代表全部在显存），正式运行中每5秒记录一次驻留。Ollama 0.35.1仅在本线程服务进程中设keep_alive=-1、num_parallel=1，系统环境未改。[热启动协议](../artifacts/eval_driven_20261006/context-online-ab/protocol-hot-v1.json)。首个S2热基线预热28.375秒（13输入/1输出Token），正式交付479.485秒、35/35通过，97次驻留观察均合格；候选596.109秒失败，未Apply或交付。候选最后三轮各编译通过并执行16项结果，15项通过、零值执行失败，后续场景未执行；重复无效修订后累计反馈35009字节超过32768字节而终止，不能写成35项全部执行或模型调用额度耗尽。SpreadAPI首个候选/基线均通过，分别376.047/437.422秒；StartupNoticeBlank首对基线/候选均通过，分别282.171/254.906秒，随后按顺序启动ConfigMerge候选。[前六次有效汇总与阶段原件](../artifacts/eval_driven_20261006/context-online-ab/summary-hot-six-started-v3_5.json)。host v3.4进一步区分有启动证据但尚无终态的incomplete与无开始证据的not_executed；既不靠状态文件声称进程live，也不推算未收尾运行的完整成本。host v3.5将读取权限失败或引用哈希不符另记evidence_unavailable，保留未知，不再误记未执行或任务失败；4项相关反例通过。早先v3.2汇总读取host目录受权限限制，测量无效原件保留，不改变受测运行结果；v3.3以具备读取权限的host只读重算，保存阶段父子跨度和原件指针，不累加嵌套时间，缺少的独立探针/人工等待/排队边界保持未知，绑定汇总器及142份实际导入项目源码。预热与交付分列且总成本不漏计，不预设优化收益。为什么要有它：启动条件改变单独版本化，失败保留在原槽位，避免把模型加载差异或筛选成功运行当成优化效果。

原条件四次探索运行均通过：S2基线/候选537.719/463.922秒，输入41,263/41,026、输出2,543/2,925 Token，实际RG执行14/11；SpreadAPI候选/基线512.031/337.187秒、均87项通过。两对耗时方向不同且启动条件不同，不能混作优化提速。旧批其余未执行项保留，未补写成绩。缓存归一记录存在0与原生计数冲突，host汇总保留两者并用原件派生已知小计；缺原生响应的调用继续未知，不把输入Token与缓存Token再次相加。

**热启动第七次与串行安排：** ConfigMerge候选385.875秒失败：首候选编译错误，修订后编译通过但175项检查有21项失败；CLI merge未接线、嵌套字典未写回及null文件被覆盖均有实际观察。上游计划未纳入Program.cs，随后累计反馈40681字节超过32768字节，新的行为反馈未进入下一轮；6/18产品调用、路由另1次，不能写成模型额度或900秒耗尽。[七次终态及第八次启动](../artifacts/eval_driven_20261006/context-online-ab/summary-hot-after-cmc1-v3_5.json)。ConfigMerge基线376.500秒也失败，最终源码与候选相同、同样21项行为失败及反馈预算终止；当前八次为基线3次通过/1次失败、候选2次通过/2次失败，[交接前八次汇总](../artifacts/eval_driven_20261006/context-online-ab/summary-hot-eight-before-rag-v3_5.json)。用户在已授权协调的另一页要求立即推进RAG实测，已核对cmb1进程退出、全部生成完成，并[明确交接模型服务](../artifacts/eval_driven_20261006/context-online-ab/service-release-after-cmb1.json)。收到明确归还前不启动第九槽或游戏，之后保持原顺序及预热条件；RAG主计时期间也不再启动pytest/build/批量扫描。在途离线pytest与host汇总分别于11:45:06及11:45:07.737 UTC前完成，该交界负载已通知RAG页如实记录。原20槽、冻结条件和全部失败不改。为什么要有它：协调发生在运行边界，不能把模型资源竞争混入耗时或重置已开始的尝试。

**C1/C2工件准备：** C1保留登记模糊题干，用户已明确房主查看全队统计；新CombatLedger scaffold以129份真实依赖首次编译通过，0警告/0错误，[Build原件](../artifacts/eval_driven_20261006/benchmark10/C1/runs/scaffold-build-v1/result.json)。统计功能、私有控制与实机尚未通过。C2已准备真实类型绑定的初始自动门工程、两种正控制、五种负控制及执行脚本；普通门与安全门不是继承关系，正常解锁和可交互是适用开门范围。C2尚未Build/实机资格通过，脚本存在不计为可开跑，[当前工件](../artifacts/eval_driven_20261006/benchmark10/C2/preparation-status.json)。

**当前开发修复（未注入冻结实验）：** C2开门子项原来可把早于候选请求或来自其他Mod/人工的Open算成功；恢复失败也可能提前解锁，测试目录还未创建。现已修正请求因果窗口、精确目录生命周期及恢复证据成功后释放锁，保留首次5失败/13通过记录，修后18/18离线反例通过；真实Build/游戏仍未跑。Ollama缓存记录另发现原生9709输入/427输出/6833缓存被旧归一写成缓存0，当前开发树正修透传与未知值语义；现有146项适配器回归通过，新测试部分接线失败已修但尚未重跑，不宣称新正常在线运行已验证。

**真实失败驱动的开发版补齐（待资源窗口复验）：** 已从ConfigMerge原始tool.response确认Program.cs曾被ArgumentException/JsonException/JsonConfigStore查询读到，却因两条需求查询对应两份材料预算而未进入Findings/Planning。当前开发版将源码材料预算与查询数解耦，并显式传递已读省略/仅命中未读事实；省略项不成为可引用证据或Apply权限，7项新反例待执行。21条编译行为失败中的重复快照/引用另以反馈内部引用表共享，保留每条检查、输入、期望、实际值和顺序，提供SHA校验还原；实现与反例已写，独立复核后将引用模块纳入worker实现身份，并补正常recovery预算/消息一致性检查；尚未执行，不声称实际预算节省或模型收益，非recovery旧路径不扩大为已压缩。缓存记录修复已补Qwen/Responses未知值、缺调查/规划回合的总量未知及规划provider_response JSON原件；48项目标检查待复验。C2 runtime-v2已补行为时序与缺证据判定，并修正不同门资格条件互相干扰及多次靠近窗口混淆；尚未运行新测试或Build。C1私有reference补全队名单和零贡献行，原readiness-v2源码已核SHA归档，[readiness-v3](../artifacts/eval_driven_20261006/benchmark10/C1/readiness-v3.json)明确21个受控断言/6种负控制仍未执行，实际名单覆盖待实机。host汇总开发版3.6已接规划provider_response原件，新增6个来源优先级/缺计数反例待执行，旧3.5脚本和结果保留。RAG主批正在另一页独占运行；双方约定主批终态后先留短验证窗口，再执行独立CPU-BGE探针。本段所有源码变更均未注入原冻结热启动、十题或RAG输入。为什么要有它：真实失败要推动实现修订，但未执行的改动不能冒充新运行成绩或覆盖旧对照。

**S3/S4受控裁判与S4中断：** S3新版37项裁判接受两种实际编译正确实现，并拦住旧33项裁判漏过的删除清理回归；S4原69项裁判接受两种实际编译正确实现，四类反例均被拒绝。两者均是私有资格控制，非Agent成绩。[S3原件](../artifacts/eval_driven_20261006/benchmark10/S3/qualification-results.json)、[S4受控原件](../artifacts/eval_driven_20261006/benchmark10/S4/judge-qualification/managed-result-v1.json)。S4正常环境首轮在加载期间相机出现后提前结束，不能算可操控关卡资格；新版增加同场次可操控确认，第二次因桌面Escape中断，确认尚未提交，关卡阶段未完成。[中断结果](C:/Users/10160/AppData/Local/Temp/s4normal_positive2/result.json)明确未通过且临时文件已恢复。S4双方模型首轮仍未启动。

**运行恢复与CLI构建：** 修复诊断初始化抛错后丢失已启动进程句柄、存活进程下提前恢复DLL的路径；保留旧源码和故障模拟记录，正式入口采用句柄接管及恢复守卫。原冻结Codex二进制已缺失；用户明确同意保持gpt-5.6-sol/medium，另记现有0.160.0的不同构建，已保存CLI及三个配套执行器，[版本与授权](../artifacts/eval_driven_20261006/benchmark10/runner-environment-v2/result.json)。原首轮快照和成绩没有覆盖。

**五任务在线对照开始：** 同一574文件当前快照、5题×两策略×两次、统一900秒，区别限查询复用/Findings压缩；原十题600秒不变。首行在GET模型目录准入时因服务未运行停止，4.984秒原件保留；服务继承的D盘模型路径不可用，已以进程级配置恢复C盘原模型缓存，digest一致。在任何生成之前另记协议补充与该额外环境尝试，首行新输出用于首次真实生成，不能把原环境故障冒充模型解题失败或抹掉成本。[协议与额外尝试](../artifacts/eval_driven_20261006/context-online-ab/protocol-v2.json)。当前尚未形成20行完成率或资源收益结论。

**10月6日补充的LLM Judge工作状态：** 自动Ragas数值评分与历史Critic/逐条语义审阅尚未形成同一条完整自动验收链；接口、评分标准或JSON结构存在不等于偏差控制、人工校准和成本优化已验证。接下来优先统一分项结论、引用证据及简短可核查理由，加入明确评分示例并用核验样本检查误判。两两比较采用时再验证顺序交换；需要替换低成本裁判时另做质量与成本对照。截图中的固定抽查比例、一致性阈值及“2%成本/90%效果”不作为本项目已验证结果或通用门禁。为什么要有它：语义诊断不能替代构建、代码行为和实际副作用验收，也不应让每次Mod任务额外等待完整模型评审流程。

**10月6日 S5 首次结果收尾：** 平台首轮459.515秒退出2，生成的候选实际编译成功，但旧受控裁判51项中17项失败，随后因行为观察超过既定反馈预算停止；未Apply、未进入正式实机验收，不能把原工程未修改误写成未生成候选。[原首轮结果](C:/Users/10160/AppData/Local/Temp/b10s5pair_20261005/results/platform/result.json)。Codex首轮连接错误后中断，保留原生会话一次源码读取调用但无返回、已知13,497 Token，完整用量、退出码和总耗时仍未知；7份源码未变，未伪造缺失result或重跑。[原件核对](../artifacts/eval_driven_20261006/benchmark10/S5/codex-interrupted/reconciliation.json)。

**10月6日逐题裁判资格：** S5原正控制仅修改观测JSON，不能证明实际修复可通过。新版本补齐实编译正控制和三个错误控制，发现并修复启动渲染线程漏查、INI跨节伪保持漏判；主线程正控制54/54通过，启动后台线程、全程后台线程和跨节伪保持均被拒绝。原缺陷在新裁判仍8项失败，原平台补丁零模型重验收仍17项失败；此次是验收补做，不是Agent续接或重跑。旧裁判、原冻结配置和首次结果保留。[实际控制、原始stdout与两版评分](../artifacts/eval_driven_20261006/benchmark10/S5/judge-qualification/managed-result-v2.json)。新实机观察器去除对允许修改的Controller私有字段及Tick方法名的依赖；真实正常环境正控制已观察到五次配置保存和SUPPLY→REPLACED画面变化，首次因退出凭证迟到及路径表示校验问题未自动通过，事后同场次补验通过，正常入口复验仍按新的运行记录判断。

**S5实机资格收尾：** 第二次运行暴露60秒局部取证窗口提前结束、总预算仍剩337秒的问题，强制退出失败原件保留。新driver-v4让终态取证使用原600秒共享截止前的剩余时间并保留20秒恢复预留；第三次私有正控制经正常入口完成五个保存窗口、终态截图、退出意图接收、owner实际观察exit0及恢复，实际裁判通过且无缺项。游戏已退出，关键文件与原插件库存检查一致。[三次记录与本次通过证据](../artifacts/eval_driven_20261006/benchmark10/S5/judge-qualification/runtime-positive-result-v1.json)。这是开发者辅助的裁判正控制、零模型调用，产品任务成功数增量0；半写/非法值/停止后效果仍由受控探针覆盖，未扩大为实机逐项覆盖。十题原冻结567文件再次核对无漂移且无pyc。

**在线资源对照准备：** 查询复用和Findings可逆压缩已具备独立开关，经正常CLI/DSH、重评及同任务续接保留实际设置并记录在manifest；默认行为保持。入口和DSH选择78项通过，尚未启动这批真实模型A/B。五题草案均已有开发或准备暴露，属于回归对照，不能称未参与调优的留出题；20次计划运行仍只有5个不同任务。[未冻结草案](../artifacts/eval_driven_20261006/context-online-ab/protocol-draft.json)。

**10月5日用户修正后的实际运行环境：** 用户明确要求停止复制/裁剪环境，直接在 `C:/Software_ Entertainment software and its platform/steamapps/common/GTFO` 保留现有 MTFO、GTFO-API、Survival 等插件测试。已确认原安装59个插件DLL、Bep控制台开启；旧Temp副本按原复制计划排除了用户plugins/patchers/config/cache，仅保留测试插件，不能代表用户完整Survival环境。旧结果、冻结模型/核心和成绩原件均保留，后续真实游戏验收按正常安装另记环境版本，不混算。

S5正常环境资格 normal1 因主机绝对路径斜杠表示误判停止（零保存动作），normal2 因观察器从首次LateUpdate计180秒导致仅完成2个保存窗口；两次失败及恢复原件保留。修正路径比较与独立session计时后，normal3在真实Survival菜单中完成5个外部保存/原子替换/重复保存窗口：每窗2506–2519ms，有真实Update→Step→Tick及GUI回调，但原始面板一直SUPPLY/96、Show更新0，正确结论是基线行为失败。0模型、0受测Agent运行；总275.516秒含人工等待和取证，不能作为Agent交付耗时。[正常环境原件和逐动作结果](../artifacts/eval_driven_20261005/benchmark10/S5/host/normal-environment-qualification/result-v2.json)。本次初始菜单截图已保存，终态截图未及时取得；半写、非法Width恢复、停止后效果与完整S5成功仍未证实。

正常环境测试均已退出并撤回测试DLL/config/control；8关键文件SHA与59原DLL路径/大小/mtime一致。未对全部用户数据做字节快照，不扩大为所有文件均未变化的声明。通用runner的selected_actions_finished/verified只证明收集完成，本题业务仍失败；不能用于任务成功计数。为什么要有它：本次正常环境的原始观察必须与业务断言绑定，收集完成和环境准备不能替代修复成功。

**10月5日候选实机反馈接线：** 新观察器 evaluator 复用全工程编译、Patch反馈与GTFO进程所有者，绑定候选源码/DLL、同一日志采样窗及部署恢复。真实旧候选188采样中174条稳定记录均未知，正确归入补丁行为失败；既有可信观察器正控制196采样、182条稳定记录均读到面板，5项通过。首次日志清单对象/数组读取错误及修正后离线复评都保留。仅证明单面板验收可区分正负，未证明完整只读、多对象或物理按键；参考控制未放入模型输入。

新独立开发ukp4沿用qwen3-coder:30b／64K／4096／180秒及18次产品额度，在488.860秒后partial：两次相同局部after_text遗漏before_text，按全binding替换会删9条using，被重复无效提交拦截。6产品＋1路由，共68,450已知Token，产品余12；没有准入候选、候选编译、Agent实机验收或Apply。已核3份初始工程文件未改、游戏退出、原观察器恢复。[本次原件、成本与资格](../artifacts/eval_driven_20261005/unity-knowledge/agent-consumption/runtime-feedback-result.json)。新验收已接线，但自动实机失败后修复的真实闭环尚未证明；不覆盖ukp1–3或冻结十题成绩。针对本次遗漏锚点，仅在native v3对应错误中补充完整替换/精确before_text的纠错说明，23项定向测试通过；未自动猜锚点、放宽校验或重跑模型证明收益。

**10月5日平台改进与新开发验证（不是十题首轮改分）：** 同一查询批次在精确源码/策略身份下复用相同调查，真实 S1 离线重放由 41 次工具执行降为 12 次，保留全部 33 条投影查询映射；修复次级命中遗漏后，已读到的 FramePanel.cs 不再被排除。首次 Findings 使用既有可逆义务文本引用，原 61,195 字节请求降为 21,561 字节，逐字段还原一致；这是输入字节证据，未宣称模型 Token 或端到端速度同比下降。[原输入与离线复核](../artifacts/eval_driven_20261005/platform-improvements/offline-s1-result.json)，最终定向核验 58 项通过。

Unity 11 份资料已独立摄取；初次长路径落盘失败与成本保留，短集合 uk1 重跑成功。修复 CRLF/BOM 原文跨度保真后，真实 36 条返回可复核；正常 Agent 又暴露“粗分类词替代实际问题”和错误码冒号被当元数据过滤的问题。当前以原始义务、已观测源码 API 名称及字面转义形成查找，真实 MCP 返回 6 项相关原文，28 项测试及嵌套错误码反例通过。[实际检索](../artifacts/eval_driven_20261005/unity-knowledge/query-context-fix/real-query-result-v2.json)。这些资料和改进未注入原冻结 benchmark。

native Patch v3 已支持同一绑定内可选精确 before_text／after_text；保留原源码、唯一锚点、范围、重叠、BOM/CRLF、多字节和 raw 模型输出核对，旧 v2/legacy 保持独立校验。独立开发案例 ukp3 的真实模型以 1,483 字节局部响应生成候选，完整编译通过，5 次产品调用＋1 次路由，已知 52,670 Token；Agent 阶段 426.266 秒。实际补验仍失败：候选把查找换成单对象 API，16 条稳定游戏观测全部 unknown:no_behaviour，且删除了多对象诊断分支。未 Apply、未交付，已停止测试游戏并恢复旧观察器；游戏补验不在该开发运行原 900 秒窗口内，不写成窗内成功。[真实生成、编译、失败与成本](../artifacts/eval_driven_20261005/unity-knowledge/agent-consumption/native-v3-result.json)。ukp1／ukp2 原 partial 分别保留；资料实际进入 provider 输入已证明，帮助完成修复尚未证明。

编译反馈到交付 adapter 的结构差异已补：依据原 compile report、worker receipt、模型/源码/DLL hash核对，33 项直接测试通过；正式8项runtime回调仍是本案例的交付条件，不能据编译通过交付上述失败候选。后续直接工作是把真实runtime反馈接入候选修订，复验实际行为。

**S4平台接线补齐（尚未开跑）：** 新工厂已实际构造，重放原六进程日志的69项断言完全一致、仍4项失败；15项定向测试通过。模型失败观察以可逆连续重复编码由855,402 UTF-8字节缩至5,789字节，保留期望、次数、顺序、类型与原执行引用；不含未来完整模型请求，未声称Token或耗时收益。[工厂与原件](../artifacts/eval_driven_20261004/benchmark10/S4/factory-ready-v4.json)。尚缺本次run路径分配、双方执行plan及真实Unity帧/相机验收；S5仍待平台接线。原十题冻结快照567文件逐项核对未漂移、无字节码缓存，Codex配置和容量失败原件未改。

**当前对照与工件边界：** 用户明确选择保留 gpt-5.6-sol／medium，先推进平台改进与其他工件；Codex S1 首轮及独立容量重试均返回 capacity，不再继续刷请求或换模型。S4 受控工件已完成；S5 3冷进程/51检查中43过、8个预期热更新失败，尚未跑受测模型或游戏验收，[S5入口](../artifacts/eval_driven_20261005/benchmark10/S5/ready-v1.json)。用户确认只有一个GTFO客户端，C5真实双端验收缺第二客户端，[环境事实](../artifacts/eval_driven_20261005/benchmark10/C5/environment-status.json)；十题分母保持10，不以单机替代联机。

**10月5日恢复后的 S1 首轮状态：** 隔离 GTFO 副本补回 413 个与原清单同哈希的缺失文件，原 12,512 文件未漂移；主菜单与基线 FPS／帧时间 HUD 已实际观察，223 个原生依赖启动后未变。只读观察器 v2 实机发现不了隐藏 BepInEx 宿主组件，失败原件保留；v3 按实际 AddComponent 宿主查找，真实 16,997 个 LateUpdate 中的 965 条采样均读到实际面板。上述均为环境资格，不是 S1 修复完成。[环境资格](../artifacts/eval_driven_20261005/benchmark10/S1/runtime-qualification/result.json)、[观察器资格](../artifacts/eval_driven_20261005/benchmark10/S1/runtime-observer-v3/qualification/result.json)。

S1 平台首次启动因准备器留下字节码缓存，在模型调用前约 2.953 秒被冻结校验拒绝，缓存与原始失败已归档。隔离缓存后另建原始源码工作区的新尝试运行 398.828 秒，第二次产品调用在 Findings 调查阶段触发既定 180 秒单请求超时；无补丁、无源码修改、未部署候选，不增加超时重刷此结果。Codex 首轮经原生连接重试后收到 provider capacity 错误，未开始工具修改；这不能归为解题失败。原始 result/Trace 保留在 `Temp/b10s1_20261005/results`，正式两臂均尚无 S1 游戏验收；[执行准备与原件](../artifacts/eval_driven_20261005/benchmark10/S1/pair-preparation/ready-final.json)。

S4 已补齐受控验收工件：6 个冷进程、69 检查中 65 通过，4 个失败均对应原始禁用日志问题；未运行受测 Agent，真实 Unity 入口仍待验证。新增 Unity 2019.4／BepInEx be665／Il2CppInterop 1.4.1 通用资料按独立版本准备，[11份资料索引](../artifacts/eval_driven_20261005/unity-knowledge/manifest-v1.json)与本批 benchmark 资料分离；实际摄取、检索和正常产品消费按后续原件更新，不以资料存在声明收益。

**10月4日十题首题S2已取得双方适用验收：** 平台原有效运行409.734秒、6次产品调用＋1次路由、34,623 tokens，候选与正式DLL均35/35；Codex在检查器环境修正后的全新原始源码副本运行245.579秒，正式35/35，仅HudSettings.cs变化且验收前后源码相同。Codex原生final用量117,770输入（含88,960缓存输入）＋1,775输出，共119,545；reasoning393为输出子项。首轮Codex600秒超时受NuGetScratch权限故障影响，原[无效对照](../artifacts/eval_driven_20261004/benchmark10/S2/pair-first/result.json)与已知失败成本保留；[修正复验](../artifacts/eval_driven_20261004/benchmark10/S2/corrected-result/result.json)明确复用平台原结果，不伪称同步重跑，不以一个样本概括总体速度或成功率。S2为新编写代表性场景，真实构造/ConfigFile/Load/布局执行，仅最终渲染接收端替换；不是历史玩家issue或实际游戏画面验收。

**下一批条件：** 其余九题沿用已冻结模型、推理、核心版本与预算。S1初始工程已编译并完成3进程/21帧基线，两个F8序列复现缺失功能；S3真实LevelAPI受控生命周期重现第二关2次/第三关3次重复提示。二者仍需相应真实游戏验证，不能把受控输入算成整题完成。用户提供的GTFO目录现在已存在GTFO.exe、GameAssembly.dll及GTFO_Data，正在核版本与隔离运行方案；尚未启动或改装主游戏目录。

**10月4日当前主线：** 按用户更新后的目标，已启动运行先收尾归档，随后优先十题真实任务对照。先推进S2非法HUD配置题的工件、适用验收与双方配置，尽快取得首题双方结果；每批固定实际执行版本，正常模型失败保留首轮结果。Reload 64K准备、澄清及经验专项不再作为开跑前置。[十题登记](../artifacts/eval_driven_20261004/benchmark10/protocol-v1.json)原登记时其余工件未冻结；最新执行结果见上方S2记录。今天同步选择四条简历主成果和三个有原件支持的案例。为什么要有它：推进顺序以真实对照和投递材料为当前产出，避免新专项持续推迟首题。

**10月4日图片错误复验：** 同一已暴露观测图经修订后的逐行描述提示、真实视觉调用和正常 query CLI，正确回答最终 Message=Latest、四次调用合计4个 Reload 事件、0条 Info 日志。1次视觉＋1次文本调用，共2,333输入／437输出 Token；视觉请求约116.58秒，查询约7.75秒、首个有效文本约5.40秒。保留原图、原始冷探针观测、冻结源码、原生请求与回答及哈希映射，见[复验原件与语义核验](../artifacts/eval_driven_20261004/image-caption-recheck/result.json)。仅证明此开发反例已通过；本次是隔离的单图稀疏查询，未重跑全量摄取或证明图片泛化收益，继承的旧增强元数据也未重新生成。

**10月4日状态与恢复验收：** 官方 MCP SDK 的真实取消通知、断连、超时、重复请求、缓存失效、不同模型权重及恶意资料/合法请求检查已实跑。复验20次生成＋4次Embedding，生成20,426输入／2,375输出 Token；旧失败、实际退出及权限边界均保留。[MCP结果](../artifacts/eval_driven_20261004/mcp-state-controls/result.json)明确区分状态通过与回答质量：部分拒答、额外权限说明及引用错误仍未消除；取消不等于上游provider立即停止。摄取内容/切块变化和重复跳过的真实记录见[原件映射](../artifacts/eval_driven_20261004/ingestion-state-control/retention.json)。

**10月4日经验学习首次四臂：** 第一题两臂均经正常入口生成、候选验收、隔离Apply/Build与正式DLL九项复验交付；均5产品＋1路由调用、相同diff。启用经验时确有一条提示进入规划，观察到产品输入／输出多848／31 Token，未见调用次数收益。[第一对证据](../artifacts/eval_driven_20261004/memory-pilot/pair-1-result.json)分列模型、正常入口与交付耗时，保留冷暖和并行干扰限制。第二题两臂原运行均partial；裁判错误地比较配置枚举顺序，已修为按键身份严格比较。保存的原始观察离线重评均12/12，见[off](../artifacts/eval_driven_20261004/memory2off-order-reassessment.json)与[on](../artifacts/eval_driven_20261004/memory2on-order-reassessment.json)，尚未恢复交付，原失败和预算不改。经验默认仍关闭，不以命中提示替代效果。

**当前未完成：** Reload组合任务仍未交付；最近开发复验仍有两项日志行为失败，后续生成被32K上下文预检阻断，18次产品额度预留8次、余10次，不能把预留数当实际发送数。历史代码无损引用与源码读取缓存的错误开关耦合已修复。澄清新复验已通过Findings/Planning，失败反馈的重复元数据已精简；cq1004c仍于第5个Patch请求被上下文预检阻断（估计33,001），产品预留8/12、余4，未交付。两轮原请求[精确重放](../artifacts/eval_driven_20261004/clarification-replay/result.json)没有新增调用、工程或文件变化。64K配置已作为单独开发策略准备保存但未执行；不改写原32K对照，也未改全局默认，按新主线暂不启动。

新增十题采用A明确需求／B症状／C模糊目标（2／6／2）、难度5简单＋5复杂，见[issue草案](../artifacts/eval_driven_20261004/benchmark10/issue-draft-v2.json)与[资料索引](../artifacts/eval_driven_20261004/benchmark10/material-index-v1.json)。门半成品复现、真实游戏版本迁移对应及同场次Host/Client双端证据仍缺；S2已完成上述首题执行；其余工件与游戏验收继续准备，未执行题保留为待测。

以下日期条目保留当时执行经过，后续结果以上方10月4日记录为准。

**10月3日完整组合实跑（进行中）：** 选定已交付 StartupNotice 的显式 ConfigFile.Reload 通知需求，冻结同源码、版本绑定 XML、真实基线观测派生图、7 个查询预期和两臂预算。首轮真实 CLI 摄取两臂均 3 文件/8 块；基础方案 8 次 embedding 请求，组合方案 3 次 batch 请求、16 次文本增强与 1 次 vision，生成模型调用的输入与输出 Token 合计 7,677，重复摄取均跳过且无新模型调用。组合首次查询因 native 改写缺发送前原件记录器而失败；修复后的真实 MCP 查询暴露 XML 清洗丢 API 标识、Dense 集合过滤筛空、URL 被当过滤器及合法拒答缺引用等问题，已修直接 owner 并补反例。新索引与查询复验继续，首轮失败、准确 caption 后回答误算总数的模型错误均保留。当前不宣称双路组合收益、完整交付或生产并发能力；CPU 采样只覆盖启动 PID，不能冒充 Python 子进程/Ollama 总峰值。

**同轮接线与学习试点：** QueryService 统一 CLI/MCP/评测，MCP 官方 SDK 实测已暴露并修复取消后的尾响应关闭竞态；`qa-tester` 入口明确完整用户目标决定验收范围。Findings 新模式区分当前规划依据与待实现行为，旧历史协议可重建；文档证据进入正常 Mod Graph 的接线仍在集成。经验试点已从真实 CS1061 失败及有效修订自动提取 `Log.Info → Log.LogInfo`，绑定 DLL 身份，并预选两项不同需求的 memory on/off 对照；四臂尚未实际运行，不宣称学习有效。Reload 的新 Mod 生成、授权交付和正式 DLL 同合同复验仍待执行。

**10月3日澄清派发复验：** d7在用户暂停时仅保留已发出的首个产品投影请求，没有原生回答或终态，Token未知；原件与派发声明保留，未自动重放。单独新试验d8中，零证据Findings改为真实missing并启动补查，但投影把“不要续跑旧任务”误当新Mod的任务控制功能，补查又重复要求新增行为已存在；6次产品调用、18,813 Token后按无新信息停止，尚有6次余量，未生成Patch或交付。两个路由调用另4,208 Token。[分轮原件与中断索引](../artifacts/eval_driven_20261002/clarification/dispatch-d8-result.json)。正在修复澄清请求表示、Findings当前知识/未来验收区分，并把既有只读依赖API检查接入正常入口；不会把工具成功返回或选路正确写成交付成功。

**10月2日恢复与新增优先项：** 用户恢复开发后，m1通过绑定原journal、完整native回答、源码前像及host停止记录接回原Product Graph。保留的第三次回答编译通过，扩充合同的51场景/72动作共175检查中154通过、21失败；失败反馈加已消耗观察达41,706字节，超过原32,768预算，未新增模型调用，仍6/12、剩6，未交付。原件未改；重复续接只读，500个文件哈希未变。[结果](../artifacts/eval_driven_20261002/interrupted_merge/result.json)。原产品与路由共28,694 Token，本次新增0。错误包括CLI未接入、递归对象变化未保存及公开API改写原null文件；不将新增验收或编译通过写成任务成功。

按用户最新顺序，当前优先补齐会话澄清续接及评测，再进行一种自动跨任务经验学习试点；工具效果随真实记录提取。澄清d1首轮为12段9段全对；d5同冻结版本开发集12段/21轮全部正确（必要澄清9/9，多余0）。首次未参与调优的留出集12段11段全对、已执行19轮18轮符合标签（必要澄清7/8）；唯一混用名称后自我纠正的句子仍有标签与模型解释分歧，原分数保留。独立审查发现的角色自相矛盾门禁已补8反例，72项相关检查通过。d6真实澄清后续接返回原交付状态，新建选路正确但Findings连续无证据支持被拒，3次产品调用/6,613 Token后阻断，尚未生成代码或交付。两例同turn重放287文件哈希不变；delivered退出码修复后正常CLI零新调用复验通过。[原始分轮成绩与成本](../artifacts/eval_driven_20261002/clarification/dispatch-first-result.json)。正在修复零证据Findings协议，旧原件保持不变。工具效果已有[只读提取与真实案例](../artifacts/eval_driven_20261002/tool_effects/result-v2.json)，区分操作完成、内容用途、同任务恢复及新增trial复验；无法确认的用途与成本保持unknown。

**9月30日上游Findings输入与检索修复：** 精确重建ap1显示查询已成功，后续Findings的29,292字节请求加4,096输出预算超出32K预检620字节。对完全相同的输入义务文本使用可逆的同消息引用后，请求降为26,631字节；全部字段可还原，源码、schema和预算不变。多词查询中的明确类型/方法名现优先匹配声明，原查询从误取ColorCrosshairDependency改为SpreadAPI。冻结源码后的真实阶段复验uv1用1次调用、5,924 Token完成证据重评并引用正确API，原ap1记录与7/12预算未改；模型编辑目标提示仍为空，本次没有执行Planning、Patch或交付。[原件与验证](../artifacts/eval_driven_20260930/upstream_context/result.json)。这是开发者触发的阶段复验，独立任务增量0，不计作原任务自动恢复或新增端到端成功。

**9月30日公开API行为修改已交付：** ap2通过正常自然语言入口为AccurateCrosshair的SpreadAPI增加负数、NaN和正负无穷的原子拒绝，只有API源文件变化，其余13个源码/配置字节不变。原DLL正常检查15/15通过、八组非法输入均暴露缺陷；worker完整编译及正式Build后的DLL各通过9个独立进程、60动作、87检查。第二轮6次产品调用＋1次路由，共31,249 Token；首轮ap1错误文件检索后请求上游证据，因上下文预检超限停止，42,660 Token及原件保留。需求显式源码标识符现加入有来源回执的检索线索，不赋予支持或写入权限。共用编译/探针检查40项、检索及直接调用者196项通过。[交付包、原件和成本](../artifacts/eval_driven_20260930/spread_api/result.json)。这是已暴露开发题，独立任务增量0；仅.NET 6冷API，真实Harmony计数并跳过游戏回调正文，未测游戏Runtime。上游上下文超限仍是独立待解失败类型，不能把本次检索成功等同于全面修复。

产品目标覆盖自然语言新建 Mod、生成中的迭代修复、接手已有问题 Mod／代码库。当前任务在[实施队列](IMPLEMENTATION_QUEUE.md)，执行边界见[产品契约](ACTIVE_PRODUCT_CONTRACT.md)；[复用审计](audits/EVALUATION_REUSE_AUDIT_20260926.md)仅供所需历史接线查询。

**9月30日同任务零模型恢复：** cv1复用fo1原生模型候选，在修复验收环境后重新编译、完成5场景/15检查，经原Graph Review/Dry-run、任务委托Apply与正式Build交付；最终DLL五场景复验通过。新增模型调用/Token均0，原产品预算仍7/12、余5；验收到Dry-run约23.27秒，交付约4.46秒。原fo1 manifest/result字节不变，首次失败仍保留；重复续接244个验收文件不变，交付后读取返回delivered。54项相关合跑、40项旧路径回归通过。[原件与回放](../artifacts/eval_driven_20260930/candidate_reverification/result.json)。独立任务增量0；仅证实这类基础设施失败后的保留候选恢复，未宣称任意故障恢复或游戏Runtime。

**9月30日已交付工程的后续需求：** fo2由自然语言选中new2的真实交付源码，新增空/纯空白Message不日志，并保留默认值、配置键、插件身份及非空日志行为。原Patch worker自动执行完整编译和5场景/15项检查，随后任务委托Apply、正式Build完成；最终DLL五场景再验通过，父DLL在两个新增场景仍产生日志，父工程未改。共7次产品调用＋1次路由、18,437 Token。首跑fo1因验收目录增加使Windows产物路径达到260字符而失败，18,445 Token及原始日志保留；缩短隔离目录后复验通过。[交付、反例和成本](../artifacts/eval_driven_20260930/followup_entry/result.json)。这是已暴露开发题，独立任务增量0；cold Load仍跳过构造器、Chainloader及游戏。68项相关Python检查和6项UI入口检查通过。fo1后续零模型恢复见上方cv1记录，历史首次失败没有改写。

**9月30日自然语言新建闭环：** 新入口`/mod`及Python `--route-request`使用一次本地模型调用选择新建、既有修改、原任务继续或澄清；六条开发用语复验6/6，首次自由理由重复导致截断的失败保留。随后“启动公告”新Mod首跑new1未交付，模型在错误日志API间重复，消耗12次预留/11次真实产品生成；加入真实CS1061接收类型对应的依赖XML说明后，new2从`Log.Info`有效修订为`Log.LogInfo`，完成完整编译、开发者配置的cold Load行为检查、原任务委托隔离Apply及正式Build，最终DLL三场景再验通过。new2为8次产品调用＋1次路由，共23,596 Token；原失败new1含路由54,072 Token。两个运行均是同一开发题，独立任务增量0；postcheck不是原worker执行，跳过构造器、由探针保存配置，未测Chainloader或游戏。源码与成本原件见[结果索引](../artifacts/eval_driven_20260930/natural_entry/result.json)。重复失败检测的临时编译路径噪声另已修复，仅有定向反例验证，不归因于new2成功。

**9月30日候选模式真实复验已交付：** 冻结当前源码后的br4经正常自然语言入口，由本地qwen3-coder:30b修复AccurateCrosshair的`Bool`编译错误并将Follow Recoil默认值改为false；完整编译、5/5真实BepInEx配置检查、精确任务委托隔离Apply及正式Build均通过。原Findings未决问题、blocked Evidence与`can_generate_patch=false`仍保留；原输入工程未改。共5/12次调用，输入/输出24,855/1,874 Token，含交付约257.15秒，见[原件索引](../artifacts/eval_driven_20260930/compilation_candidate_product/result.json)。这是已暴露的开发夹具，独立任务增量0，未测游戏Runtime及自动LiveEdit；一次成功不证明稳定完成率。br1服务失败、br2/br3在Patch前停止的原记录保留。此前离线58项整合检查与17项既有journal回归见[收尾记录](../artifacts/eval_driven_20260930/candidate_closeout/closeout.json)。其后新建与生成修订的真实结果见上方9月30日自然语言新建闭环。

当前采用[真实交付与评测共同推进](EVALUATION_FIRST_CYCLE.md)：交付过程产生验收、成本和边界记录，具体短板驱动改动，候选效果由新任务检验。七类待测、企业问题与实际可填数据集中在[成果清单](PROJECT_OUTCOMES.md)。

**本地模型已交付一个已暴露的配置修改需求。** AccurateCrosshair 在 `dev07_definition` 和公共 runner 的 `dev08_entry` 两个版本运行中均完成自然语言入口→补丁→**5/5** 配置验收→任务委托 Apply→正式 Build；同题开发者控制与原失败分别保留。另有两道独立工程题，首轮基线 0/2、候选 1/2，但三轮累计仅基线0/6次、候选1/6次成功；交付尚不稳定。Transfer 的尾部空白硬拒及续接缺口已修，进入真实行为验收后仍仅4/8，当前失效点为无效修订。原件、成本、边界及后续处理依据统一见[结果记录](audits/EVALUATION_FIRST_RESULTS_20260926.md)。

| 能力 | 已确认事实 | 尚未由这些事实证明 |
| --- | --- | --- |
| 正常产品入口 | `cli.run-request` → `local_entry.run_local_request` → 原 Graph / 共享 Patch worker；可经 worker 配置注入行为裁判 | 任意任务、项目或语言都可完成 |
| 真实阶段评测 | `run_patch_sft_eval.execute_prepared` 已有真实 provider、C# 行为裁判和修订路径 | 给定 Plan 的结果不是自然语言到交付的成功率 |
| 组合验收 | `CompleteCrosshairVerification` 组合编译、配置和适用行为矩阵；通过现有动态工厂配置接入 | 配置/冷行为矩阵不等于游戏 Runtime |
| 授权与续接 | 原人审路径兼容；`task_delegation_decision_v1` 绑定精确 diff/context、原源码和通过的验收，closure3 已实际 Apply/Build；专项检查覆盖重复消费及漂移 | 任意崩溃、跨环境恢复；将任务委托记成人审或产品模型自主成功 |
| Trace 与用量 | 两次独立失败过程重建2/2；嵌入usage已接通，18次检索复验639输入Token；六次配置交付成本完整 | 人工定位提速；本地AI定位对照尚无正确配对 |
| 独立双臂 | 两题三轮、共12次真实运行；基线0/6次、候选1/6次成功；冻结任务、代码、模型、权限、预算及裁判身份一致 | 更多独立任务与稳定交付；独立性仍受已核本地暴露范围限定 |
| 文档 RAG | 3份RFC/90块/14留出题；BM25与Hybrid引用正确率16/24、11/20，正确拒答均3/3，来源辅助AI审阅 | 广语料、代码导航或端到端任务收益 |
| 上下文策略 | 相同配置任务三对均交付；去重＋复用输入66,655→134,337，平均工具4→7；已回退普通入口开关、保留校准修复 | 该组合策略的资源收益；更多任务上的泛化 |
| 回归与门禁 | 原47文件本地1420通过、2跳过；当前选集53文件，测量与CLI139通过、诊断修订相关42项分批通过 | 新版53文件完整执行、远程CI及独立产品任务的自动发布准入 |

这些是分层事实，不将源码存在、历史通过和本次执行混写。为什么要有它：短状态页需要可信，同时不能靠省略限制扩大成果。

## 2. 从能力找到实际接线

需要定位接线时可使用现有只读导航，例如：

```powershell
.venv/Scripts/python.exe -B -m scripts.engineering_cognition --kind symbol --name CompleteCrosshairVerification --pretty
.venv/Scripts/python.exe -B -m scripts.engineering_cognition --kind symbol --name RequestPatchExecution --pretty
```

根据当前问题沿以下路径核对，不把每次维护都变成全仓扫描。为什么要有它：静态符号引用可加速定位，但动态工厂和实际验收仍需追到具体入口。

| 层次 | 定位 |
| --- | --- |
| 入口 | [CLI](../src/mod_agent/cli.py) 的 `run-request` 与 [local_entry](../src/mod_agent/local_entry.py) 的 `run_local_request` |
| 配置 | `--patch-worker-config` → [PatchWorkerConfiguration](../src/mod_agent/patch_worker_configuration.py) |
| 工厂 | `behavior_evaluator.factory_ref / factory_name / kwargs` → [make_complete_crosshair_evaluator](../src/mod_agent/domain/complete_crosshair_verification.py) |
| 执行 | [RequestPatchExecution](../src/mod_agent/request_patch_execution.py) → 共享 worker 与配置的行为 evaluator；原 Graph 继续拥有审批和执行 |
| 验收 | [组合验收测试](../tests/unit/test_complete_crosshair_verification.py)、[公开产品入口测试](../tests/unit/test_mod_agent_cli.py)及所选真实任务判定 |

## 3. 当前缺口与版本边界

Transfer阶段控制曾达公开3/4；正常入口td6/td7现已结束，两臂均未交付，诊断增加调用而仍重复失败代码，默认保持关闭。下一步按真实任务与证据缺口选择工作，Transfer不作为所有后续任务的前置条件。文档RAG保留BM25默认；Self-RAG微调及外部Critic尚无采用收益。源码观察组合优化已回退；定位对照两臂均0/2，停止同配置跑分。Jev仍只作[方案评估](JEV_FEASIBILITY_20260928.md)。数字和采用依据统一见[成果清单](PROJECT_OUTCOMES.md)，原始经过保留在[结果记录](audits/EVALUATION_FIRST_RESULTS_20260926.md)。

主树包含跨任务未提交实现；运行或并行修改前检查受影响文件及工作区身份，不以 HEAD 单独代表当前代码，不为隔离任务丢弃已有 WIP。为什么要有它：现成入口与验收脚本可能尚未提交，错误起点会漏掉真实能力。

[评测协议](EVALUATION_DRIVEN_DEVELOPMENT.md)供相应指标和实验计数使用，[交付与评测方式](EVALUATION_FIRST_CYCLE.md)说明当前推进方法，原候选另存于[历史计划](EVALUATION_FIRST_CYCLE.history-20260927.md)。新开发可调整实现与合同，已运行实验仍按原协议保留分母、预算和结果。为什么要有它：历史记录应帮助判断，不能成为新交付的不可变开发门禁。

## 4. 历史记录

此前状态、完整测试命令、任务额度及日期叙述保存在[2026-09-26 字节归档](PROJECT_STATE.history-20260926.md)。归档身份与本次验证见[清理记录](audits/ACTIVE_ENTRY_CLEANUP_20260926.md)。

<a id="follow-recoil-baseline-2026-09-14"></a>
历史链接兼容：原 [Follow Recoil baseline 2026-09-14](PROJECT_STATE.history-20260926.md#follow-recoil-baseline-2026-09-14) 已归档，不是当前任务。
