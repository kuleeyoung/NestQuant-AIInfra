# M1 split-K mixed-length saturation probe

This diagnostic used 32 deterministic bimodal requests at a mean offered load of
4 req/s, with prompt lengths 4-48 and output lengths 2-12. It was stopped before
the third candidate run after the first two complete pairs showed that queueing and
cross-process state dominated the comparison.

Observed TPOT P95 ranged from 241 ms to 2405 ms and TTFT P95 from 4507 ms to
20914 ms. Reversing policy order also reversed which process was faster, so these
runs cannot attribute a speedup or regression to split-K. They establish only that
this long-prompt mix exceeds the useful local capacity of the current engine setup.

The incomplete batch is retained under `results/raw/m1_splitk_mixed_ab_v1/` as
diagnostic evidence and is excluded from the promotion result. The replacement
low-load paired experiment is reported in
`results/w4a4_m1_down_splitk_mixed_ab_v1.md`.
