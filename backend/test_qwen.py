from app.ai.inference import qwen


print("Starting Qwen test...")

result = qwen.interpret(
    vendor="Cisco",
    platform="IOS/IOS-XE",
    section="global configuration",
    previous_lines=[
        "version 17.9",
        "hostname AI-TEST-RTR",
    ],
    current_line="aaa new-model",
    next_lines=[],
)

print("\n================ QWEN RESULT ================\n")
print(result)
print("\n================================================\n")