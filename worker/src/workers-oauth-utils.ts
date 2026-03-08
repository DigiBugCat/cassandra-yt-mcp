import type { AuthRequest, ClientInfo } from "@cloudflare/workers-oauth-provider";

export class OAuthError extends Error {
  constructor(
    public code: string,
    public description: string,
    public statusCode = 400,
  ) {
    super(description);
    this.name = "OAuthError";
  }

  toResponse(): Response {
    return new Response(
      JSON.stringify({
        error: this.code,
        error_description: this.description,
      }),
      {
        status: this.statusCode,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

export interface OAuthStateResult {
  stateToken: string;
}

export interface ValidateStateResult {
  oauthReqInfo: AuthRequest;
  clearCookie: string;
}

export interface BindStateResult {
  setCookie: string;
}

export interface CSRFProtectionResult {
  token: string;
  setCookie: string;
}

export interface ValidateCSRFResult {
  clearCookie: string;
}

export function sanitizeText(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function sanitizeUrl(url: string): string {
  const normalized = url.trim();
  if (normalized.length === 0) {
    return "";
  }
  for (let index = 0; index < normalized.length; index += 1) {
    const code = normalized.charCodeAt(index);
    if ((code >= 0x00 && code <= 0x1f) || (code >= 0x7f && code <= 0x9f)) {
      return "";
    }
  }

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(normalized);
  } catch {
    return "";
  }

  const scheme = parsedUrl.protocol.slice(0, -1).toLowerCase();
  return ["https", "http"].includes(scheme) ? normalized : "";
}

export function generateCSRFProtection(): CSRFProtectionResult {
  const csrfCookieName = "__Host-CSRF_TOKEN";
  const token = crypto.randomUUID();
  return {
    token,
    setCookie: `${csrfCookieName}=${token}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=600`,
  };
}

export function validateCSRFToken(formData: FormData, request: Request): ValidateCSRFResult {
  const csrfCookieName = "__Host-CSRF_TOKEN";
  const tokenFromForm = formData.get("csrf_token");
  if (!tokenFromForm || typeof tokenFromForm !== "string") {
    throw new OAuthError("invalid_request", "Missing CSRF token in form data", 400);
  }

  const cookieHeader = request.headers.get("Cookie") || "";
  const cookies = cookieHeader.split(";").map((value) => value.trim());
  const csrfCookie = cookies.find((value) => value.startsWith(`${csrfCookieName}=`));
  const tokenFromCookie = csrfCookie ? csrfCookie.substring(csrfCookieName.length + 1) : null;
  if (!tokenFromCookie) {
    throw new OAuthError("invalid_request", "Missing CSRF token cookie", 400);
  }
  if (tokenFromForm !== tokenFromCookie) {
    throw new OAuthError("invalid_request", "CSRF token mismatch", 400);
  }

  return {
    clearCookie: `${csrfCookieName}=; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=0`,
  };
}

export async function createOAuthState(
  oauthReqInfo: AuthRequest,
  kv: KVNamespace,
  stateTTL = 600,
): Promise<OAuthStateResult> {
  const stateToken = crypto.randomUUID();
  await kv.put(`oauth:state:${stateToken}`, JSON.stringify(oauthReqInfo), {
    expirationTtl: stateTTL,
  });
  return { stateToken };
}

export async function bindStateToSession(stateToken: string): Promise<BindStateResult> {
  const consentedStateCookieName = "__Host-CONSENTED_STATE";
  const hashHex = await hashValue(stateToken);
  return {
    setCookie: `${consentedStateCookieName}=${hashHex}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=600`,
  };
}

export async function validateOAuthState(
  request: Request,
  kv: KVNamespace,
): Promise<ValidateStateResult> {
  const consentedStateCookieName = "__Host-CONSENTED_STATE";
  const url = new URL(request.url);
  const stateFromQuery = url.searchParams.get("state");
  if (!stateFromQuery) {
    throw new OAuthError("invalid_request", "Missing state parameter", 400);
  }

  const storedDataJson = await kv.get(`oauth:state:${stateFromQuery}`);
  if (!storedDataJson) {
    throw new OAuthError("invalid_request", "Invalid or expired state", 400);
  }

  const cookieHeader = request.headers.get("Cookie") || "";
  const cookies = cookieHeader.split(";").map((value) => value.trim());
  const consentedStateCookie = cookies.find((value) =>
    value.startsWith(`${consentedStateCookieName}=`),
  );
  const consentedStateHash = consentedStateCookie
    ? consentedStateCookie.substring(consentedStateCookieName.length + 1)
    : null;
  if (!consentedStateHash) {
    throw new OAuthError(
      "invalid_request",
      "Missing session binding cookie - authorization flow must be restarted",
      400,
    );
  }

  if ((await hashValue(stateFromQuery)) !== consentedStateHash) {
    throw new OAuthError(
      "invalid_request",
      "State token does not match session - possible CSRF attack detected",
      400,
    );
  }

  let oauthReqInfo: AuthRequest;
  try {
    oauthReqInfo = JSON.parse(storedDataJson) as AuthRequest;
  } catch {
    throw new OAuthError("server_error", "Invalid state data", 500);
  }

  await kv.delete(`oauth:state:${stateFromQuery}`);

  return {
    oauthReqInfo,
    clearCookie: `${consentedStateCookieName}=; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=0`,
  };
}

export async function isClientApproved(
  request: Request,
  clientId: string,
  cookieSecret: string,
): Promise<boolean> {
  const approvedClients = await getApprovedClientsFromCookie(request, cookieSecret);
  return approvedClients?.includes(clientId) ?? false;
}

export async function addApprovedClient(
  request: Request,
  clientId: string,
  cookieSecret: string,
): Promise<string> {
  const approvedClientsCookieName = "__Host-APPROVED_CLIENTS";
  const thirtyDaysInSeconds = 2592000;
  const existingApprovedClients = (await getApprovedClientsFromCookie(request, cookieSecret)) || [];
  const updatedApprovedClients = Array.from(new Set([...existingApprovedClients, clientId]));
  const payload = JSON.stringify(updatedApprovedClients);
  const signature = await signData(payload, cookieSecret);
  const cookieValue = `${signature}.${btoa(payload)}`;
  return `${approvedClientsCookieName}=${cookieValue}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=${thirtyDaysInSeconds}`;
}

export interface ApprovalDialogOptions {
  client: ClientInfo | null;
  server: {
    name: string;
    logo?: string;
    description?: string;
  };
  state: Record<string, any>;
  csrfToken: string;
  setCookie: string;
}

export function renderApprovalDialog(request: Request, options: ApprovalDialogOptions): Response {
  const { client, server, state, csrfToken, setCookie } = options;
  const encodedState = btoa(JSON.stringify(state));
  const serverName = sanitizeText(server.name);
  const clientName = client?.clientName ? sanitizeText(client.clientName) : "Unknown MCP Client";
  const serverDescription = server.description ? sanitizeText(server.description) : "";
  const logoUrl = server.logo ? sanitizeText(sanitizeUrl(server.logo)) : "";
  const clientUri = client?.clientUri ? sanitizeText(sanitizeUrl(client.clientUri)) : "";
  const policyUri = client?.policyUri ? sanitizeText(sanitizeUrl(client.policyUri)) : "";
  const tosUri = client?.tosUri ? sanitizeText(sanitizeUrl(client.tosUri)) : "";
  const contacts =
    client?.contacts && client.contacts.length > 0
      ? sanitizeText(client.contacts.join(", "))
      : "";
  const redirectUris =
    client?.redirectUris && client.redirectUris.length > 0
      ? client.redirectUris
          .map((uri) => {
            const validated = sanitizeUrl(uri);
            return validated ? sanitizeText(validated) : "";
          })
          .filter((uri) => uri !== "")
      : [];

  const htmlContent = `
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${clientName} | Authorization Request</title>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background-color: #f8fafc;
            margin: 0;
            padding: 0;
          }
          .container {
            max-width: 600px;
            margin: 2rem auto;
            padding: 1rem;
          }
          .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.1);
            padding: 2rem;
          }
          .header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
          }
          .logo {
            width: 48px;
            height: 48px;
            margin-right: 1rem;
            border-radius: 8px;
          }
          .client-info {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
          }
          .client-detail {
            margin-bottom: 0.5rem;
          }
          .actions {
            display: flex;
            justify-content: flex-end;
            gap: 1rem;
            margin-top: 1.5rem;
          }
          .button {
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
          }
          .button-primary {
            background: #0f766e;
            color: white;
          }
          .button-secondary {
            background: white;
            color: #1f2937;
            border: 1px solid #d1d5db;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="card">
            <div class="header">
              ${logoUrl ? `<img src="${logoUrl}" alt="${serverName} logo" class="logo">` : ""}
              <h1>${serverName}</h1>
            </div>
            ${serverDescription ? `<p>${serverDescription}</p>` : ""}
            <h2><strong>${clientName}</strong> is requesting access</h2>
            <div class="client-info">
              <div class="client-detail"><strong>Name:</strong> ${clientName}</div>
              ${clientUri ? `<div class="client-detail"><strong>Website:</strong> ${clientUri}</div>` : ""}
              ${policyUri ? `<div class="client-detail"><strong>Privacy:</strong> ${policyUri}</div>` : ""}
              ${tosUri ? `<div class="client-detail"><strong>Terms:</strong> ${tosUri}</div>` : ""}
              ${redirectUris.length > 0 ? `<div class="client-detail"><strong>Redirect URIs:</strong> ${redirectUris.join("<br>")}</div>` : ""}
              ${contacts ? `<div class="client-detail"><strong>Contact:</strong> ${contacts}</div>` : ""}
            </div>
            <form method="post" action="${new URL(request.url).pathname}">
              <input type="hidden" name="state" value="${encodedState}">
              <input type="hidden" name="csrf_token" value="${csrfToken}">
              <div class="actions">
                <button type="button" class="button button-secondary" onclick="window.history.back()">Cancel</button>
                <button type="submit" class="button button-primary">Approve</button>
              </div>
            </form>
          </div>
        </div>
      </body>
    </html>
  `;

  return new Response(htmlContent, {
    headers: {
      "Content-Security-Policy": "frame-ancestors 'none'",
      "Content-Type": "text/html; charset=utf-8",
      "Set-Cookie": setCookie,
      "X-Frame-Options": "DENY",
    },
  });
}

async function getApprovedClientsFromCookie(
  request: Request,
  cookieSecret: string,
): Promise<string[] | null> {
  const approvedClientsCookieName = "__Host-APPROVED_CLIENTS";
  const cookieHeader = request.headers.get("Cookie");
  if (!cookieHeader) return null;
  const cookies = cookieHeader.split(";").map((value) => value.trim());
  const targetCookie = cookies.find((value) => value.startsWith(`${approvedClientsCookieName}=`));
  if (!targetCookie) return null;
  const cookieValue = targetCookie.substring(approvedClientsCookieName.length + 1);
  const parts = cookieValue.split(".");
  if (parts.length !== 2) return null;

  const [signatureHex, base64Payload] = parts;
  const payload = atob(base64Payload);
  const isValid = await verifySignature(signatureHex, payload, cookieSecret);
  if (!isValid) return null;

  try {
    const approvedClients = JSON.parse(payload);
    if (!Array.isArray(approvedClients) || !approvedClients.every((item) => typeof item === "string")) {
      return null;
    }
    return approvedClients as string[];
  } catch {
    return null;
  }
}

async function hashValue(value: string): Promise<string> {
  const hashBuffer = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(hashBuffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function signData(data: string, secret: string): Promise<string> {
  const key = await importKey(secret);
  const enc = new TextEncoder();
  const signatureBuffer = await crypto.subtle.sign("HMAC", key, enc.encode(data));
  return Array.from(new Uint8Array(signatureBuffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function verifySignature(signatureHex: string, data: string, secret: string): Promise<boolean> {
  const key = await importKey(secret);
  const enc = new TextEncoder();
  try {
    const signatureBytes = new Uint8Array(
      signatureHex.match(/.{1,2}/g)!.map((byte) => Number.parseInt(byte, 16)),
    );
    return await crypto.subtle.verify("HMAC", key, signatureBytes.buffer, enc.encode(data));
  } catch {
    return false;
  }
}

async function importKey(secret: string): Promise<CryptoKey> {
  if (!secret) {
    throw new Error("cookieSecret is required for signing cookies");
  }
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { hash: "SHA-256", name: "HMAC" },
    false,
    ["sign", "verify"],
  );
}
