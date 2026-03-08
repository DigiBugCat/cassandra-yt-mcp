# Cassandra YT MCP Infra Modules

These modules are owned by the `cassandra-yt-mcp` service so the Worker, backend auth
contract, and Cloudflare resource shape can evolve together.

Environment roots live in `cassandra-infra` and are expected to compose:

- `cassandra-yt-mcp//infra/modules/worker-edge`
- `cassandra-yt-mcp//infra/modules/backend-access`
- `cassandra-infra//modules/cloudflare-tunnel`

## Modules

- `modules/worker-edge`
  - Worker DNS record
  - KV namespace for OAuth state
  - WAF skip rule for MCP traffic
- `modules/backend-access`
  - Cloudflare Access application and service token for the tunneled backend hostname

## Expected Deploy Order

1. Apply the environment root from `cassandra-infra`
2. Deploy the k8s backend and cloudflared tunnel
3. Publish the Worker from `worker/` with Wrangler
