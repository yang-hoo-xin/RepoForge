"""Regression cases for vague-only requests versus words inside specific requests."""
import pytest
from src.mod_agent.planning import RequestPlanner

@pytest.mark.parametrize("user_request", [
    "Fix the outdated PlayerAgent Hook using C:/sdk/LatestMonoDlls/Modules.dll.",
    "Add a test that verifies disabled HUD logging stays quiet.",
    "Fix it: the door opens twice after entering the level.",
    "修一下击杀提示，重新进关后每次击杀都会显示两次。",
    "那个问题是配置读取失败，请让非法HUD大小恢复默认值。",
    "请修复更新后失效的Hook，并保持帧计数；API路径是R6LatestMonoDlls。",
])
def test_specific_request_is_not_blocked_by_incidental_vague_phrase(user_request):
    plan = RequestPlanner().plan(user_request, target_mod="ExistingMod")
    assert not plan.blocked
    assert not any(u.uncertainty_id == "ambiguous_user_request" for u in plan.uncertainties)

@pytest.mark.parametrize("user_request", [
    "fix it", "Please fix it.", "Could you fix that issue?", "test please",
    "请帮我处理一下这个东西", "麻烦你帮忙修一下那个问题吧", "修一下", "test",
])
def test_vague_only_request_still_requests_clarification(user_request):
    plan = RequestPlanner().plan(user_request, target_mod="ExistingMod")
    assert plan.blocked
    assert plan.route == "blocked_ambiguous"
    assert any(u.uncertainty_id == "ambiguous_user_request" for u in plan.uncertainties)
