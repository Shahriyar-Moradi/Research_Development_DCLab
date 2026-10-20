from .extensions import optional

cli = optional("cli") or optional("open_cli")  # the research CLI in the full edition, the studio's `dclab` command in the open one
raise SystemExit(cli.main())
