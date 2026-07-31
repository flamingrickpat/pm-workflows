# pm-workflows

`pm-workflows` is the reusable workflow kernel. It loads a workflow manifest,
dispatches isolated role sessions, runs declared checks, routes declared
outcomes, journals attempts, and manages git checkpoints.

The import package is `pm_workflows`. The distribution installs `pm-coder` from
the GitHub repository.

Workflow metadata is stored outside the target repository in
`~/.pm/pm-workflows/<timestamp>_<task-id>/`.
