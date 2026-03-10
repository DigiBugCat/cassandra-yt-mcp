import type { McpAuthEnv } from "cassandra-mcp-auth";

declare global {
  // Extend McpAuthEnv with service-specific bindings via interface merging
  interface Env extends McpAuthEnv {
    BACKEND_BASE_URL: string;
    BACKEND_API_TOKEN?: string;
    CF_ACCESS_CLIENT_ID?: string;
    CF_ACCESS_CLIENT_SECRET?: string;
  }
}

export {};
