from transformers import RagRetriever, AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from datasets import load_dataset
import torch

# 1. 데이터셋 로드 및 검색용 문서 설정
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")

# 문서 ID와 실제 텍스트를 매핑하기 위한 데이터셋 구축
doc_id_to_text = {str(i): doc["article"] for i, doc in enumerate(dataset)}

# 2. LLaMA 생성기 준비 (생성 작업 담당)
llama_model_name = "meta-llama/Llama-3.2-1B"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

# 3. RAG 검색기 준비 (RAG의 문서 검색 기능 사용)
rag_model_name = "facebook/rag-sequence-nq"
rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
retriever = RagRetriever.from_pretrained(
    rag_model_name,
    index_name="exact",           # 정확한 일치 검색
    passages_path=None,           # 사전 구축된 문서 경로가 없으면 None
    use_dummy_dataset=False       # 실제 데이터셋 사용
)

# 4. 질의 인코딩을 위한 DPRQuestionEncoder 준비
dpr_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_encoder_name)
dpr_encoder = DPRQuestionEncoder.from_pretrained(dpr_encoder_name)

# 5. 입력 질의
query = "What is the significance of renewable energy?"

# 6. 질의 인코딩 (DPR을 사용하여 question_hidden_states 생성)
input_ids_query = dpr_tokenizer(query, return_tensors="pt")["input_ids"]

with torch.no_grad():
    question_hidden_states = dpr_encoder(input_ids_query).pooler_output

# 텐서를 numpy 배열로 변환
question_hidden_states = question_hidden_states.cpu().numpy()

# 7. 문서 검색 (input_ids_query와 question_hidden_states를 모두 retriever로 전달)
input_ids_query = input_ids_query.cpu().numpy()  # 텐서를 numpy 배열로 변환
retrieved_docs = retriever(input_ids_query, question_hidden_states)

# 8. 검색된 문서 ID로 실제 문서 텍스트를 가져오기
retrieved_doc_ids = retrieved_docs["doc_ids"]

# 디버깅 출력: 검색된 문서 ID를 출력
print("Retrieved Document IDs:", retrieved_doc_ids)
print("Available document IDs in doc_id_to_text:", list(doc_id_to_text.keys())[:10])  # 일부분만 출력

# 9. 문서 ID 배열에서 개별 ID를 추출하여 실제 문서 텍스트를 가져옴
retrieved_text = []
for doc_id_array in retrieved_doc_ids:
    for doc_id in doc_id_array:
        doc_id_str = str(doc_id)  # 문서 ID를 문자열로 변환
        if doc_id_str in doc_id_to_text:
            retrieved_text.append(doc_id_to_text[doc_id_str])  # 문서 텍스트를 추가
        else:
            print(f"Document ID {doc_id_str} not found in doc_id_to_text")

# 디버깅 출력: 검색된 텍스트 확인
print("Retrieved Texts:", retrieved_text)

# 10. LLaMA 모델로 생성 작업 수행
for doc in retrieved_text:
    # 검색된 문서를 LLaMA에 입력해 텍스트 생성
    inputs = llama_tokenizer(doc, return_tensors="pt")
    with torch.no_grad():
        outputs = llama_model.generate(**inputs)
    
    # 생성된 텍스트 출력
    generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text from retrieved doc: {generated_text}")
