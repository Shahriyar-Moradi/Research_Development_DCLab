from dclab_rnd.memory.reading import memory_off_for_evaluation

memory_off_for_evaluation()  # 35.4: a graded run never reads a remembered line

from .run import main

raise SystemExit(main())
