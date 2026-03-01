# Aegis

**Independent verification layer for autonomous AI agent purchases.**

Aegis intercepts AI agent purchase intents, runs multi-dimensional verification checks, and returns an approve/flag/block decision — before money moves.

> Your AI agent is about to spend $347 on the wrong hotel. Aegis catches that.

## Why Aegis?

AI agents are increasingly making autonomous purchases — booking travel, ordering supplies, executing transactions. But there's no standard infrastructure to verify these purchases before they execute.

Agents overpay. They buy the wrong thing. They fall for scam sellers. They agree to hidden subscriptions. Aegis fixes this.

## Quick Start
```bash
pip install aegis-verify
```
```python
from aegis import verify_purchase

result = await verify_purchase(
    intent={
        "item": "Sony WH-1000XM5 Headphones",
        "price": 278.00,
        "seller": "electronics-deals-store.com",
        "original_instruction": "best noise canceling headphones under $300"
    }
)

if result.decision == "approve":
    await execute_purchase()
elif result.decision == "flag":
    print(f"Concerns: {result.reasons}")
elif result.decision == "block":
    print(f"Blocked: {result.reasons}")
```

## What Aegis Checks

| Module | What It Does |
|--------|-------------|
| Price Verification | Compares against market prices across multiple sources |
| Intent Matching | Verifies purchase aligns with the original user request |
| Authorization | Checks budgets, spending limits, and permissions |
| Seller Verification | Validates merchant legitimacy and reputation |
| Terms Review | Analyzes refund policies, auto-renewals, hidden fees |

## Integrations

### LangChain
```python
from aegis.integrations import LangChainAuditTool
tools = [LangChainAuditTool()]
```

### CrewAI
```python
from aegis.integrations import CrewAIAuditTool
agent = Agent(tools=[CrewAIAuditTool()])
```

### Claude MCP
```json
{
  "mcpServers": {
    "aegis": {
      "command": "python",
      "args": ["/path/to/aegis_mcp_server.py"]
    }
  }
}
```

Install and run:
```bash
pip install aegis-verify "mcp[cli]"
python aegis_mcp_server.py
```

## How It Works
```
User Request → Agent Proposes Purchase → Aegis Verifies → Approve/Flag/Block → Execute or Halt
```

See the full [Developer Workflow Diagram](docs/workflow.html) for details.

## Try the Demo

See Aegis in action with three real-world scenarios — a good purchase, a sketchy deal, and a rogue agent:

```bash
git clone https://github.com/CurtisRearden/aegis.git
cd aegis
pip install aegis-verify
python demo.py
```

The demo runs all five verification modules against each scenario and shows the approve/flag/block decision with detailed scoring and reasons — all in milliseconds, before any money moves.

## Roadmap

- [x] Core verification engine with 5 modules
- [x] LangChain integration
- [x] CrewAI integration
- [x] Claude MCP server
- [ ] Hosted API with dashboard
- [ ] Seller-side verification tools
- [ ] Decentralized validator network

## Contributing

Contributions welcome! Open an issue or submit a pull request.

## License

MIT — see [LICENSE](LICENSE) for details.
