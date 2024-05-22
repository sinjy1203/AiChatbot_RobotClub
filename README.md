# RobotClub Chatbot
- 인하대학교 로보트연구회 동아리에 대한 정보를 알려주는 AI 챗봇
- 카카오톡을 통해 챗봇에 동아리에 대한 정보를 입력해주면 챗봇이 정보들을 바탕으로 질문에 답해준다
## Architecture
![chatbot_robotclub drawio](https://github.com/sinjy1203/AiChatbot_RobotClub/assets/52316531/b8e87302-e2f2-43c0-8f65-60c3c143b16d)
## Prerequisites
- `Docker`, `kubernetes`, `helm` installed
## Usage
### 1. Clone code
```bash
git clone https://github.com/sinjy1203/AiChatbot_RobotClub.git
```
### 2. Prepare model(LLM, Embedding model)
```bash
docker run --gpus all -v {model_directory}:/prepare_model/models sinjy1203/prepare-model
```
### 3. Server 실행
- helm/values.yaml 에서 맞는 디렉토리 수정
```bash
cd helm
# langsmith 사용 안함
helm install ai-chatbot-robot-club .
# langsmith 사용
helm install --set langchain_tracing_v2=t --set langchain_api_key={langsmith_api_key} ai-chatbot-robot-club .
```
