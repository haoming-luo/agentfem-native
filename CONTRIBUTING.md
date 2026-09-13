# Contributing

AgentFEM Native currently accepts changes only through the clean-room process.

Each contribution must include:

1. a real requirement and capability maturity label;
2. mathematical specification and algorithm sources;
3. tests, failure cases, and convergence evidence appropriate to the claim;
4. a provenance declaration stating whether AI was used and whether any
   third-party finite-element implementation was viewed;
5. confirmation of contributor identity and agreement to the project CLA;
6. native Windows, macOS, and Linux evidence for platform-sensitive changes.

Do not contribute copied, translated, rearranged, or AI-paraphrased code from
FEniCSx or another finite-element implementation. Do not include code with an
unclear license or ownership chain.

The project owner records CLA acceptance before merging an external
contribution.

重要代码注释、公开说明、数学规格、ADR 和验证报告以中文为主。稳定代码
标识符、JSON 字段、ABI 和错误码保留英文，并提供中文含义与诊断。完整规则见
`docs/development/CHINESE_DOCUMENTATION_POLICY.md`。
