# reporting

Owner:coordinator. Source:the user implementation assignment and PDF E/H operational requirements.

Export actual analysis records,all window statuses and prescribed-strength curves.

Public interfaces:`generate_report(analysis,destination)`.

Implementation:[reporting.py](../../../src/operations/reporting.py). Verification:[operations tests](../../../tests/unit/operations/test_operations.py). Deployment:[runbook](../../runbook.md). Local fixtures verify failure and integrity behavior; actual OAuth,provider,GPU,cloud lifetime and full experiment checks remain separately gated.

```mermaid
---
title: "reporting operational implementation"
---
flowchart TD
    INPUT["Validated configuration and immutable inputs"] --> CALL["src/operations/reporting.py"]
    CALL --> VERIFY["Export actual analysis records;all window statuses and prescribed-strength curves"]
    VERIFY --> OUT["Actual verified / failed / blocked evidence"]
```
