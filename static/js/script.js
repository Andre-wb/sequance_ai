    async function predict() {
        const input = document.getElementById("sequence").value;
        const numbers = input.split(",").map(num => num.trim());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sequence: numbers })
            });

            const result = await response.json();
            if (response.ok) {
                document.getElementById("output").innerHTML = `
                            <p>Продолжение: ${result.predictions.map(n => n.toFixed()).join(", ")}</p>
                        `;
            } else {
                document.getElementById("output").innerHTML = `<p style="color: red;">Ошибка: ${result.error}</p>`;
            }
        } catch (error) {
            document.getElementById("output").innerHTML = `<p style="color: red;">Произошла ошибка: ${error.message}</p>`;
        }
    }
