
from email.message import EmailMessage
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr

import json, os
import smtplib

ACCOUNTS_FILE = "account.json"

router = APIRouter(
    prefix = '/login',
    tags = ['LOGIN'],
)


class LoginForm(BaseModel):
    username: str
    password: str


class SignupForm(BaseModel):
    email: EmailStr
    password: str


def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return {}
    with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_accounts(data):
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def send_permission_email(to_email: str, username: str, base_url: str):
    link = f"{base_url}/login/signup/permit/{username}"
    subject = "[AutoML] 帳號申請通知"
    body = f"""
    嗨，敬愛的大大：

    Email: {to_email} 的帳號申請已收到。

    點擊以下連結啟用帳號：
    {link}
    """

    #send_email(to_email, subject, body)
    send_email( 'Stephen.Chen@primax.com.tw', subject, body )

def send_permitted_notification_email(to_email: str):
    subject = "[AutoML] 帳號啟用成功"
    body = f"""
    您已被許可登入 AutoML Platform：
    http://xxxx
    """

    send_email(to_email, subject, body)


def send_email(to_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "noreply@primax.com.tw"  # 改為你的發信帳號
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP("10.40.1.119", 25) as smtp:
            smtp.send_message(msg)
    except Exception as e:
        print(f"⚠️ Email 發送失敗: {e}")



@router.post( '' )
def login( form: LoginForm ) :
    accounts = load_accounts()

    # 檢查帳號
    login_key = form.username
    user = accounts.get(login_key)
    if not user:
        for key, info in accounts.items():
            if info["email"].lower() == form.username.lower():
                user = info
                login_key = key
                break
    if not user:
        raise HTTPException(status_code=404, detail="Username not found")

    # 檢查密碼
    if user["password"] != form.password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    # 登入成功
    return {
        "message": "Login successful",
        "username": login_key,
        "email": user["email"]
    }

@router.post( '/signup' )
def signup(form: SignupForm, request: Request):
    #if form.password != form.confirmPassword:
    #    raise HTTPException(status_code=400, detail="Passwords do not match")

    username = form.email.split("@")[0]

    accounts = load_accounts()

    # 建立/更新帳號，以 username 為 key
    accounts[username] = {
        "email": form.email,
        "password": form.password,
        "permit": False
    }

    save_accounts(accounts)

    # 取得 base URL（用來產生驗證連結）
    base_url = str(request.base_url).rstrip("/")

    # 發送 Email
    send_permission_email(form.email, username, base_url)

    return {"message": "Signup successful", "account": accounts[username]}



@router.get("/signup/permit/{username}")
def permit_user(username: str):
    accounts = load_accounts()

    if username not in accounts:
        raise HTTPException(status_code=404, detail="User not found")

    accounts[username]["permit"] = True
    save_accounts(accounts)

    user_email = accounts[username]["email"]
    send_permitted_notification_email(user_email)

    return {"message": f"Account '{username}' is now permitted."}
