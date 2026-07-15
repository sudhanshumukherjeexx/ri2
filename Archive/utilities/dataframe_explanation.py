def AnalyzeDataFrame(df, client, custom_prompt):
    # Convert DataFrame to CSV string
    csv_string = df.to_csv(index=False)
    
    # Create a chat completion using the OpenAI GPT-4 model
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant asked to analyze data."
            },
            {
                "role": "user",
                "content": custom_prompt
            },
            {
                "role": "user",
                "content": csv_string
            }
        ],
        max_tokens=1024,
        model="gpt-4o"
    )  
    return chat_completion