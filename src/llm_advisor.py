import ollama


def get_maintenance_advice(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    prediction,
    confidence,
):
    """
    Generate a maintenance recommendation using Llama 3.
    """

    prompt = f"""
You are an industrial predictive maintenance engineer.

Analyze the following machine information together with the machine learning prediction and provide a maintenance recommendation.

Machine Information:
- Machine Type: {machine_type}
- Air Temperature: {air_temperature} K
- Process Temperature: {process_temperature} K
- Rotational Speed: {rotational_speed} rpm
- Torque: {torque} Nm
- Tool Wear: {tool_wear} min

Machine Learning Result:
- Prediction: {prediction}
- Confidence: {confidence:.2f}%

Instructions:

If the prediction is "Machine Failure":
- Treat the situation as critical.
- Explain what the sensor values may indicate.
- Recommend immediate inspection and corrective maintenance.
- Advise stopping the machine before further operation if appropriate.

If the prediction is "No Machine Failure":
- State that the machine appears to be operating normally.
- Recommend only routine preventive maintenance.
- Do not recommend stopping the machine.

Respond using exactly this format:

Summary:
<One short sentence>

Possible Cause:
<One or two short sentences. Use expressions such as "may indicate", "could suggest", or "is consistent with". Do not present uncertain causes as facts.>

Recommended Action:
- <Action 1>
- <Action 2>
- <Action 3>

Rules:
- Keep the response under 100 words.
- Do not repeat the input sensor values.
- Be concise, technical, and professional.
- Do not use markdown.
- Do not mention that you are an AI or language model.
"""

    # Send prompt to Llama 3
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Return generated recommendation
    return response["message"]["content"]