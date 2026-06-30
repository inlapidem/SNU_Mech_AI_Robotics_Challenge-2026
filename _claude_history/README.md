# Claude Code 채팅 내역 백업

이 폴더는 `~/.claude/projects/-home-user-joon/` 의 대화 기록(.jsonl)과 메모리입니다.

## 새 컴퓨터에서 채팅 내역 복원하기
1. 이 repo를 clone 합니다. **가능하면 같은 경로 `/home/user/joon` 에 두세요.**
2. 아래로 복사합니다:
   ```bash
   mkdir -p ~/.claude/projects/-home-user-joon
   cp -r _claude_history/* ~/.claude/projects/-home-user-joon/
   ```
3. 프로젝트 경로가 `/home/user/joon` 과 다르면, 폴더 이름 `-home-user-joon` 을
   새 경로에 맞게 바꿔야 합니다. 예: 경로가 `/home/kim/joon` 이면 → `-home-kim-joon`
   (절대경로의 `/` 를 `-` 로 바꾼 이름)
