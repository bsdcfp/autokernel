# H200 Codex 宿主升级

2026-09-20，用户授权升级 CLI 和模型。**升级已执行并确认 `codex-cli 0.155.1`；模型配置已更新，实际账户可用性待登录后验证。** 此目录仅保存不含凭据的配置，禁止加入 auth.json、API key 或登录日志。

- 环境入口：`/aigc/engineering/fuping.chu/env/codex_env.sh`。
- 共享 npm prefix：`/aigc/engineering/fuping.chu/.npm-global`。
- Codex CLI：`@openai/codex@0.132.0` → npm 当前稳定版 `@openai/codex@0.155.1`。
- 模型：`gpt-5.5` → `gpt-6-astra`，保持 `medium`，不改其他配置项。
- 原配置完整备份：`config.before.toml`；待部署配置：`config.toml`。只在远端内容与原备份一致时替换，避免覆盖并发编辑。
- 新 npm 包 integrity：`sha512-02fAAGyBtlA1zPjEo3kTj/bOSYbPz5DvjLwRZJdV7weFFEDzNFOMjQGmZ/+5CuirYV0hE+AZTrnjzwXYU4AdAQ==`。
- 官方源码仓库：<https://github.com/openai/codex>；包通过 npm 官方 registry 获取。
- 回退：加载同一环境后安装 `@openai/codex@0.132.0`，将已备份的 `config.before.toml` 恢复至共享 CODEX_HOME/config.toml。登录凭据不涉及本次配置替换。

依据：[官方模型迁移说明](https://developers.openai.com/api/docs/guides/latest-model)、[官方登录说明](https://learn.chatgpt.com/docs/auth)。配置更新不等于认证成功；登录后还须核对账户实际可用模型。

AutoKernel 上游代码、候选、评测器与门限均不属于此次升级范围。

原配置在部署前已备份至 GitHub 提交 `fea8c06`；远端另保留 `config.before-20260920.toml`。已按原文件字节一致性校验后原子替换配置。

用户在 Pod 内登录：

```bash
source /aigc/engineering/fuping.chu/env/codex_env.sh
codex login --device-auth
codex login status
```
