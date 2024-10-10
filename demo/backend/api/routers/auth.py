import io
import json
import os
import sys
from typing_extensions import Annotated
from jose.exceptions import JWTError
from jose import jwt
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.security import (
    HTTPBasic,
    HTTPBasicCredentials,
    OAuth2PasswordRequestForm,
    OAuth2PasswordBearer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"

from services.auth_service import create_access_token
from utils.hashing import get_password_hash, verify_password

from api.schemas.users import User, UserInDB
from db.database import db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")
auth_router = r = APIRouter()
security = HTTPBasic()


@r.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await db.users.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@r.post("/register")
async def register(username: str, password: str):
    # Check if the user already exists
    existing_user = await db.users.find_one({"username": username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # If the user does not exist, proceed with registration
    hashed_password = get_password_hash(password)
    user = {"username": username, "hashed_password": hashed_password}
    await db.users.insert_one(user)
    return {"username": username, "hashed_password": hashed_password}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user


@r.get("/authorization", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user
