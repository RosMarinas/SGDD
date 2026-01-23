# GEMINI.md


## Core Philosophy

1. 你的修改应该使得项目更易于理解和维护，而不是更复杂。需要进行改进时请直接在源代码中修改，而不是新建文件;
2. 在开始或完成某个功能前，请先在plan.md中记录你的计划和思路，确保团队成员了解你的工作内容;
3. 你可以编写脚本临时测试功能，但是请新建一个tests/目录，将这些脚本放在那里，并在完成后删除它们;
4. 请尽量复用已有代码和函数，避免重复造轮子，不要新建总结文件，所有总结和说明请直接写在GEMINI.md或plan.md中;
5. 在完成某个任务后请及时更新plan.md与本文件，确保它们与当前代码状态一致，不要新建总结文件.
6. issues中记录的bug和改进建议请及时处理，并在解决后在issues中给出回应.
7. 用户可能会修改代码，请你在覆盖用户修改前，先检查用户的改动内容，确保不会覆盖用户的重要修改.
## 环境
+ 使用uv作为包管理器，pyproject.toml中有一些现成的库，请通过修改pyproject.toml来添加新的依赖。
+ 请使用uv run 来运行代码


## Reference Files

- `plan.md` - Detailed implementation plan (in Chinese)
- `GEMINI.md` - This file