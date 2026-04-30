import bcrypt

passwords = {
    "admin": "rahma@admin28",
    "user":  "rahma@user28",
}

for username, pwd in passwords.items():
    hashed = bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print(f"{username}: {hashed}")