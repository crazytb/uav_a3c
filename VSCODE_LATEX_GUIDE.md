# VS Code LaTeX 설정 가이드

VS Code에서 LaTeX 논문을 컴파일하고 편집할 수 있도록 환경이 설정되었습니다.

## 1. 확장 프로그램 설치

VS Code에서 다음 확장 프로그램을 설치하세요:

1. **LaTeX Workshop** (james-yu.latex-workshop) - 필수
2. **LTeX** (valentjn.vscode-ltex) - 문법 검사 (선택사항)

### 자동 설치
프로젝트를 열면 VS Code가 권장 확장 프로그램 설치를 제안합니다.

## 2. 컴파일 방법

### 방법 1: 자동 컴파일 (권장)
- `paper.tex` 파일을 저장하면 자동으로 컴파일됩니다
- 설정: `"latex-workshop.latex.autoBuild.run": "onSave"`

### 방법 2: 수동 컴파일
1. **Command Palette** (`Cmd+Shift+P`)
2. `LaTeX Workshop: Build LaTeX project` 검색 및 실행

### 방법 3: 단축키
- `Ctrl+Alt+B` (Windows/Linux) 또는 `Cmd+Option+B` (macOS)

### 방법 4: 사이드바 버튼
- 왼쪽 사이드바에서 LaTeX 아이콘 클릭
- Build LaTeX project 버튼 클릭

## 3. PDF 보기

컴파일이 완료되면:
- VS Code 내부 탭에서 PDF가 자동으로 열립니다
- 실시간 동기화: LaTeX 소스와 PDF 간 양방향 동기화 지원

## 4. 빌드 옵션

### 사용 가능한 레시피:
1. **pdflatex → bibtex → pdflatex × 2** (기본, 참고문헌 포함)
2. **pdflatex** (빠른 컴파일, 참고문헌 제외)

### 레시피 변경:
Command Palette → `LaTeX Workshop: Build with recipe` → 원하는 옵션 선택

## 5. 추가 기능

### 자동 정리
- 컴파일 후 `.aux`, `.log` 등 임시 파일 자동 삭제
- 수동 정리: Command Palette → `LaTeX Workshop: Clean up auxiliary files`

### IntelliSense
- LaTeX 명령어 자동완성
- 수학 기호 지원
- 참고문헌 자동완성

### 에러 처리
- 컴파일 에러 시 Problems 패널에 표시
- 에러 위치로 바로 이동 가능

## 6. 단축키

| 기능 | 단축키 (macOS) | 단축키 (Win/Linux) |
|------|----------------|-------------------|
| 빌드 | `Cmd+Option+B` | `Ctrl+Alt+B` |
| 정리 | `Cmd+Option+C` | `Ctrl+Alt+C` |
| 동기화 | `Cmd+Option+J` | `Ctrl+Alt+J` |

## 7. 문제 해결

### 컴파일이 안 될 때:
1. Problems 패널에서 에러 확인
2. 출력 패널에서 LaTeX Workshop 로그 확인
3. Command Palette → `LaTeX Workshop: View Log`

### PDF가 업데이트되지 않을 때:
1. Command Palette → `LaTeX Workshop: Clean up auxiliary files`
2. 다시 빌드

### 한글 폰트 문제:
- XeLaTeX 사용 시: `\usepackage{kotex}` 추가
- 필요시 설정에서 빌드 도구를 xelatex로 변경

## 8. 고급 설정

현재 설정된 주요 옵션들:
- 저장 시 자동 빌드
- 빌드 후 임시 파일 자동 정리
- PDF 탭 내부 표시
- 에러/경고 메시지 표시

설정 수정: `.vscode/settings.json` 파일 편집