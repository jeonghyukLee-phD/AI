1. 가상환경 생성 
```bash
python3 -m venv myenv
```
2. 가상환경 활성화
```bash
source myenv/bin/activate
```
3. 선행모듈 설치
```bash
pip install torch transformers datasets accelerate sentencepiece protobuf peft
```
4. Hugging Face Access token 받기
a. 모델 레포지토리에 접근하려면, 예를 들어, LLaMA-2-7b-hf 페이지(https://huggingface.co/meta-llama/Llama-2-7b-hf)에서 Access Request 버튼을 클릭해 접근 권한을 요청합니다.
b. Hugging Face Token 설정 페이지-Access Tokens에 접속하여 Create New Token 버튼을 클릭합니다. 이후 토큰을 생성한 후, 사용하려는 레포지토리에 접근 권한이 추가되었는지 확인합니다.

5. Hugging Face login
```bash
huggingface-cli login
```
6. 스크립트 실행
```bash
python src/"fileName".py
```
