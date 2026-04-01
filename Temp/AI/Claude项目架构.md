---
date: 2026-04-01
author:
  - Siyuan Liu
tags:
  - summary
---
## 整体目录结构

```
my-project/
├── CLAUDE.md                    # 项目级指令（提交到 git）
├── CLAUDE.local.md              # 个人本地指令（不提交）
├── .mcp.json                    # MCP 服务器配置（提交到 git）
│
└── .claude/
    ├── settings.json            # 项目共享设置（提交到 git）
    ├── settings.local.json      # 个人本地设置（不提交，自动加入 .gitignore）
    │
    ├── skills/                  # 技能/自定义命令
    │   ├── review/
    │   │   └── SKILL.md
    │   ├── deploy/
    │   │   ├── SKILL.md
    │   │   └── scripts/
    │   └── testing-patterns/
    │       └── SKILL.md
    │
    └── rules/                   # 按路径作用域的规则（大项目用）
        ├── frontend.md
        └── backend.md

~/.claude/                       # 全局用户级（跨所有项目）
├── CLAUDE.md                    # 全局个人指令
├── settings.json                # 全局用户设置
├── settings.local.json          # 全局本地设置
├── .credentials.json            # 认证凭据（勿提交）
├── commands/                    # 全局斜线命令（旧版，现合并入 skills）
│   ├── review.md
│   └── commit.md
└── projects/                    # 各项目的会话历史
    └── -Users-you-myproject/
        └── session-xxx.jsonl
```

---

## 各文件详解

### `CLAUDE.md` — 项目记忆核心

Claude 启动时自动读取，相当于给 AI 的项目入职指南。典型内容：

```markdown
# Project Overview
电商后端 API，基于 FastAPI + PostgreSQL。

# Architecture
- src/api/       — 路由层
- src/services/  — 业务逻辑
- src/models/    — 数据模型

# Commands
- `make dev`     — 启动开发服务器
- `make test`    — 运行测试
- `make lint`    — 代码检查

# Coding Conventions
- 所有函数必须有类型注解
- 禁止裸 except，必须指定异常类型

# Critical Gotchas
- DB 查询必须用 projection，禁止 SELECT *
```

CLAUDE.md 可以放在多个位置：`~/.claude/CLAUDE.md` 对所有会话生效；项目根目录 `./CLAUDE.md` 提交 git 与团队共享；父目录适合 monorepo；子目录的 CLAUDE.md 在处理该目录文件时按需加载。

---

### `settings.json` vs `settings.local.json`

配置作用域优先级从高到低：Managed（企业管理）> User（用户全局）> Project（项目共享）> Local（本地个人）。

```json
// .claude/settings.json（提交 git，团队共享）
{
  "$schema": "https://schemas.anthropic.com/claude-code/settings.json",
  "permissions": {
    "allow": ["Bash(npm run *)", "Bash(make *)"],
    "deny": ["Bash(rm -rf *)", "Read(.env)"]
  },
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{ "type": "command", "command": "eslint $FILE", "timeout": 10 }]
    }]
  }
}
```

```json
// .claude/settings.local.json（不提交，个人偏好）
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  }
}
```

---

### `.claude/skills/` — 技能系统

在 `.claude/skills/` 中创建带 SKILL.md 的目录来赋予 Claude 领域知识和可复用工作流。相关时 Claude 自动应用，也可用 `/skill-name` 直接调用。Custom commands 已合并进 skills 系统，两者现在等价。

```
.claude/skills/
├── check-score/        ← 你刚做的那个
│   └── SKILL.md
└── code-review/
    ├── SKILL.md
    └── scripts/
        └── run-checks.sh
```

---

### `.mcp.json` — MCP 服务器

放在项目根目录，提交到 git 供团队共享，用于连接 JIRA、GitHub、Slack、数据库等外部工具。

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./src"]
    }
  }
}
```

---

## 哪些文件该不该提交 git

|文件|提交？|原因|
|---|---|---|
|`CLAUDE.md`|✅ 是|团队共享项目上下文|
|`.claude/settings.json`|✅ 是|团队共享权限规则|
|`.claude/skills/`|✅ 是|团队共享工作流|
|`.mcp.json`|✅ 是|团队共享工具配置|
|`CLAUDE.local.md`|❌ 否|个人偏好|
|`.claude/settings.local.json`|❌ 否|自动加入 `.gitignore`|
|`~/.claude/.credentials.json`|❌ 绝对不|认证凭据|