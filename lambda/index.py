# lambda/index.py
import json
import os
import re  # 正規表現モジュールをインポート
import urllib.request
from urllib.error import URLError, HTTPError


# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# 推論APIのURL（Google Colab上のFastAPIエンドポイント）
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "https://b5e2-35-245-86-16.ngrok-free.app/generate")

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # FastAPI用のリクエストペイロードを構築
        request_payload = {
            "model_id": MODEL_ID,
            "messages": messages
        }
        
        print("Calling FastAPI inference endpoint with payload:", json.dumps(request_payload))
        
        # FastAPIエンドポイントにリクエストを送信
        req = urllib.request.Request(
            INFERENCE_API_URL,
            data=json.dumps(request_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                response_body = json.loads(response.read().decode('utf-8'))
                print("Inference API response:", json.dumps(response_body, default=str))
                
                # 応答の検証
                if not response_body.get('success') or not response_body.get('response'):
                    raise Exception("No valid response from the inference API")
                
                # アシスタントの応答を取得
                assistant_response = response_body['response']
                
                # アシスタントの応答を会話履歴に追加
                messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                # 成功レスポンスの返却
                return {
                    "statusCode": 200,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                        "Access-Control-Allow-Methods": "OPTIONS,POST"
                    },
                    "body": json.dumps({
                        "success": True,
                        "response": assistant_response,
                        "conversationHistory": messages
                    })
                }
        
        except HTTPError as e:
            print(f"HTTPError: {e.code} {e.reason}")
            raise Exception(f"HTTPError: {e.code} {e.reason}")
        except URLError as e:
            print(f"URLError: {e.reason}")
            raise Exception(f"URLError: {e.reason}")
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
