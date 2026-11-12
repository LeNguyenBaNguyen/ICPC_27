# Worked trajectory: source inspection, not independent human annotation

The AI assistant inspected the archived source for `F126`, aiohttp `WebSocketResponse._post_start` in `aiohttp/web_ws.py`. This is the hand-picked case already used in Figure 1. It is not a random validation sample, a human adjudication, or evidence of overall judge accuracy.

At anchor `7d0f980af6d725970cda5cb011400f5459699301`, the relevant source is:

```python
self._reader = FlowControlDataQueue(request._protocol, 2**16, loop=loop)
request.protocol.set_parser(
    WebSocketReader(self._reader, self._max_msg_size, compress=self._compress)
)
request.protocol.keep_alive(False)
```

At touch 1 (`5241897f...`), the constructor call changes to:

```python
self._reader = WebSocketDataQueue(request._protocol, 2**16, loop=loop)
```

The `keep_alive(False)` call remains. The assistant can verify these lexical facts from the stored snapshots. This does not independently establish all runtime semantics of either queue or the protocol API.

| Recorded claim | Anchor verdict | Later observations | Analysis endpoint |
| --- | --- | --- | --- |
| Claim 6: initializes a queue using `FlowControlDataQueue` | supported | contradicted at touch 1; not judged afterwards | First detection at touch 1 |
| Claim 8: disables HTTP keepalive | supported | supported at recorded touches 1–8 | Right-censored after observed follow-up; no contradiction detected |

The stored censor reason is `head`; neither claim in this example has an unresolved interval or a removal event. Do not add those states to this trajectory. Unresolved cases and removal occur elsewhere in the archive and have different handling:

- An anchor-contradicted claim is excluded from the survival cohort; it is an initial judge verdict, not a later detection event.
- An anchor-supported claim later marked unresolved stays under observation in the detection-time analysis. Unresolved is not treated as established truth. The separate sensitivity analysis censors before that touch or treats it as an event.
- An unmatched removal/rename ends observation and is censored in the primary endpoint; the calendar competing-risk analysis treats removal separately. It does not supply a verdict about semantic truth.
- After a first contradiction, subsequent correctness or restoration is unknown under alive-only judging. No resurrection check has been performed.

A full trajectory checked by independent annotators remains outstanding. Use `VALIDATION.md` for the prepared blinded audit.
