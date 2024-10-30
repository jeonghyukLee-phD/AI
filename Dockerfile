# 1. Ubuntu 베이스 이미지 선택
FROM ubuntu:20.04

# 2. 기본 패키지 업데이트 및 Python 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \ 
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파일 복사
COPY src/ /app/src
COPY requirements.txt .
COPY .env .

# 5. Python 가상환경 생성
RUN python3 -m venv venv

# 6. 가상환경에 패키지 설치
# 가상환경을 활성화하지 않고, 직접 경로를 사용해 pip 실행
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# 7. 환경 변수 설정하여 가상환경이 기본 Python 환경이 되도록 설정
ENV PATH="/app/venv/bin:$PATH"

# 8. 컨테이너가 시작될 때 기본적으로 bash 셸로 접근
CMD ["/bin/bash"]
