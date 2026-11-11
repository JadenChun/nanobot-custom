# A3-7 implementation packet — Mobile pairing and remote-access cutover

**Status.** Ready for packet review. Dispatch after A3-5 passes. Qualify with the A3-6 capability and resource limits before release.
[Delegation cutover](A3-5-delegation-cutover.md) · [Remote findings](../module-review-2026-11-06/reports/R01.md) · [Mobile implementation contract](../../mobile-client-implementation.md)

## Objective and boundary

Pair a phone to the current desktop owner with a device key.
Keep `RemoteAccessOwner`, private owner IPC, `MobileGatewayEdge`, Tailscale Serve, native tsnet, and device revocation.
Use one versioned credential path after cutover.
Keep credentials, signing, and transport in the native owner. Use the shared React app for the visible pairing and chat surfaces.

## 1. Cut over the pairing and access protocol — R01, R03, R11

The desktop owner creates a short-lived one-use offer after it verifies the private Serve mapping.
Use the same offer lifecycle for QR and manual code. Keep the existing collection limits and expiry checks.
The phone creates a non-exportable Android Keystore signing key. Send the public key and installation identity during collection.
Bind the pending record to the offer, instance, expected TLS origin, installation, and key fingerprint.
Show the bound device on the desktop. Approval authorizes that pending record only.
Persist enough pending state on both sides to recover the same unexpired collection after restart.
Native Cancel revokes the local pending generation and clears its persisted state.
Reject a late collect or approve response for a cancelled, denied, or superseded attempt.

For access issuance, use a fresh server challenge bound to the approved device and its current generation.
P-256 with SHA-256 is the proposed signing choice.
Approve the algorithm, key and signature encoding, verifier, and canonical signed bytes before implementation.
Supply matching Android and server fixtures for valid, expired, replayed, and wrong-origin proofs.
Include protocol version, purpose, instance and origin binding, installation ID, challenge ID, nonce, and expiry.
Verify the current device and consume the challenge before issuing a short-lived scoped access token.
When the token response is lost, the phone can request a new challenge with the same key.
Keep signing keys and bearer tokens outside the WebView.
[Android Keystore](https://developer.android.com/privacy-and-security/keystore)

**Edit boundary.** `nanobot/remote/pairing.py`, `nanobot/remote/auth.py`, `nanobot/remote/store.py`, `nanobot/remote/owner.py`, and the owner routes in `nanobot/remote/edge.py`.
On Android use the key and pending-state paths in `mobile/android/app/src/main/java/ai/nanobot/mobile/network/CustodyStore.kt` and `mobile/android/app/src/main/java/ai/nanobot/mobile/network/NativeNetworkPlugin.kt`.
Record the challenge consumption point and v1 revocation and rollback behavior before implementing issuance.

**AC7.1.** QR and manual code pair exactly the approved device.
Expiry, denial, and Cancel leave no usable pending authority.
Both phone and owner restart can recover a still-valid attempt. A cancelled attempt stays cancelled.
Replayed, expired, wrong-key, or wrong-origin proofs cannot issue a token.
Lost access-token delivery can recover by fresh proof. Revocation prevents further issuance and use.

## 2. Bind routes and deployment to the owner — R03, R16

Resolve the private owner socket and state from the A3-1 instance root.
Verify an exact Tailscale Serve target and expected TLS host. Suppress the offer if that mapping cannot be verified.
Use the native tsnet resolver and dialer for tailnet transport. Keep TLS server-name verification through the route.
Use the production dispatcher as the authority for the operation inventory.
Bind HTTP and WebSocket reads, attachment commands, and mutations to explicit action scopes after identity revalidation.
Deny unknown commands and revoked scopes at the dispatch boundary.
Register or replace the pairing-renew action and route together in the versioned contract.
Require effective authentication for any non-loopback WebUI listener.
Use authenticated mutation admission for the local desktop notification side effect.
[Tailscale Serve](https://tailscale.com/kb/1242/tailscale-serve) · [Authority findings](../module-review-2026-11-06/integrated/I01-authority.md)

**Edit boundary.** `nanobot/remote/composition.py`, `nanobot/remote/tailscale.py`, `nanobot/remote/protocol.py`, and `nanobot/remote/operations.py`.
Keep mobile scope checks in the production edge and shared command dispatch in `nanobot/webui/inbound_commands.py`.
Use `nanobot/webui/gateway_endpoint.py`, `nanobot/webui/ws_http.py`, and the existing remote settings routes for their own admission checks.
Update the versioned operation inventory and generated clients through `mobile/scripts/build-operations-contract.py`.

**AC7.2.** A valid reduced-scope credential can use its declared operations only.
The production route inventory matches the negotiated version.
An incorrect Serve target, wrong TLS origin, unauthenticated non-loopback listener, or cross-origin notification attempt fails closed.
Two explicit gateways cannot consult each other's owner socket or state.

## 3. Settle native requests and share the pairing UI — R04, R11, R13

Install the request correlation before native work can complete.
Retain one bounded terminal result for each admitted request, even when the normal event queue is full.
Bound request count and frame size before admission, encoding, and JNI delivery.
Account for JSON and base64 expansion. Reject an oversize frame before it can block the event pump.
Return native errors by request ownership. Keep signed media URLs and credentials out of logs.
Reserve cache bytes before writing large media. Bypass caching when capacity is unavailable and continue the permitted stream.

Use a typed native bootstrap state for unpaired, pending, paired, and failed results.
Render the pairing screen through the existing shared React app. Mount the normal app after pairing.
Retain only the native bridge and packaging bootstrap needed to enter that flow.

**Edit boundary.** `mobile/native-network/internal/broker/`, `mobile/native-network/cmd/lib/`, and the owning Android network and media adapters.
Use `mobile/mobile-bootstrap.js`, `webui/src/lib/mobile-runtime.ts`, and the existing shared React startup path for the client cutover.

**AC7.3.** An immediate completion and a full queue still settle the correct request once.
Oversize input and concurrent native failures cannot stall the bridge or cross request identities.
Cache pressure stays within the reserved bytes. Logs contain no signed media URLs or credentials.

## 4. Cut over and qualify

Mark v1 bearer devices legacy and revoked before enabling v2 issuance.
Tell old clients that the server requires an upgrade. Require re-pairing for the new device-key contract.
Do not preserve a live bearer-refresh authority alongside device proof.
Qualify the gateway, production edge, command router, native bridge, and shared UI together.

**AC7.4.** A physical Android device pairs, restarts, loses and recovers access delivery, and revokes through the real desktop owner.
Exercise Tailscale re-login, MagicDNS unavailable to the host OS, missing Serve, and wrong origin.
An old app receives the version failure. A v1 refresh credential cannot regain access after cutover.

Start with `tests/remote/test_pairing.py`, `tests/remote/test_auth.py`, `tests/remote/test_edge_integration.py`, `tests/remote/test_owner_ipc.py`, `tests/remote/test_tailscale.py`, and `tests/webui/test_mobile_commands.py`.
Use the broker tests, Android pairing and custody instrumentation, and the shared client pairing identity checks.
Label fixture-only, emulator, and physical evidence separately.
Keep revocation state during rollback. If a rollback build lacks the v2 authority, disable remote pairing and access until a compatible build is restored.
Return the v2 wire and persisted-state compatibility evidence to A3-9.
