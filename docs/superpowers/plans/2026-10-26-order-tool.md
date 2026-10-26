# Order query tool implementation plan

**Goal:** Implement the order tool approved in this chat; leave prompts for the user.

**Architecture:** The support endpoint forwards its Authorization header as per-run
AgentContext. query_order uses Uplink/aiohttp to call the Java gateway and returns
only paidAt, status, payableAmountFen and paidAmountFen. Amounts stay in fen;
unpaid values stay null. Do not branch on Result.code.

**Files:** app/clients/java.py, app/schemas/agent_context.py, app/tools/order.py,
app/services/agent_service.py, app/api/agent.py; tests/test_order_tool.py and
tests/test_agent_context.py.

**Review focus:** Hidden JWT argument; concurrent request isolation; no JWT in
messages/Redis; no recipient information returned; HTTP failures and timeout.

- [x] Write and run failing tool and request-context tests.
- [x] Implement the tool and support-only context forwarding. Use
  MALL_GATEWAY_BASE_URL, defaulting to the Java gateway's localhost:8080.
- [x] Run focused tests, full pytest suite, Ruff and inspect the final diff.

Verification: baseline 28 passed; new tests failed before implementation;
focused suite 24 passed; full suite 45 passed with two existing dependency
deprecation warnings; Ruff and git diff --check passed. HTTP tests use only
synthetic JWTs and a temporary loopback gateway, not real Java business data.
The publication snapshot, excluding unrelated local Nacos edits, passed all
43 tests with the same two dependency deprecation warnings.

The Java client sets Authorization once as a session default. Each tool call
uses an independent session. context_schema is omitted; astream still passes
the per-run AgentContext containing the current Authorization header.

The user authorized implementation and push to GitHub. Preserve unrelated Nacos
edits and existing knowledge documentation locally; publish only this order tool
change. Leave prompt editing to the user and make no Java source changes.
