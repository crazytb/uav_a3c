# LaTeX Paper Environment

IEEE Transaction format 논문 작성 환경이 설정되었습니다.

## 파일 구조

```
.
├── paper.tex              # 메인 논문 파일 (IEEE transaction 형식)
├── references.bib         # 참고문헌 BibTeX 파일
├── compile.sh            # 논문 컴파일 스크립트
├── figures/              # 그래프 및 이미지 파일들
└── sections/             # 논문 섹션별 tex 파일들
    ├── introduction.tex
    └── methodology.tex
```

## 사용법

### 논문 컴파일
```bash
./compile.sh
```

### 임시 파일 정리
```bash
./compile.sh clean
```

### 수동 컴파일 (필요시)
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## 논문 구조

현재 논문은 다음과 같은 구조로 설정되어 있습니다:

1. **Abstract** - 연구 요약
2. **Introduction** - 연구 배경 및 동기
3. **Related Work** - 관련 연구
4. **Methodology** - 연구 방법론
5. **Experimental Setup** - 실험 설정
6. **Results and Analysis** - 결과 및 분석
7. **Discussion** - 논의
8. **Conclusion** - 결론

## 분석 결과 활용

기존의 분석 결과들을 논문에 포함시키려면:

1. `figures/` 폴더에 분석 이미지들을 복사
2. `paper.tex`에서 `\includegraphics{figures/filename}` 사용
3. 결과 분석 섹션에 통계적 분석 내용 추가

## IEEE 형식 특징

- IEEEtran 클래스 사용
- 2단 컬럼 레이아웃
- IEEE 표준 참고문헌 형식
- 적절한 수학 패키지 포함