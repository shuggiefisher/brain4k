# What is GitBrains?


GitBrains is a framework that intends to improve the reproducability and
shareability of machine learning models.  Think of it as MVC for machine learning,
except with GitBrains pipelines you have Data, Models, and Metrics.

The rules of GitBrains:
1. GitBrains live in version control, and so can be forked, reverted and managed like other code.
2. Each stage in the pipeline is deterministic and reproducible for a given commit
3. GitBrains is framework and language agnostic - pipe from one language to another if your execution environment supports it
4. Every pipeline publishes performance metrics to encourage quality models to rise to the top
5. Contribute plugins for your preferred ML framework back to the community