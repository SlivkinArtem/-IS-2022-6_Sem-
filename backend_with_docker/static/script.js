let lastGeneratedLyrics = ""; // для хранения последнего текста

async function likeLyrics() {
    if (!lastGeneratedLyrics) return;

    try {
        const response = await fetch("/like", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: lastGeneratedLyrics })
        });

        const data = await response.json();
        alert(data.message || "Сохранено!");
    } catch (err) {
        console.error("Ошибка при сохранении:", err);
        alert("Ошибка при сохранении.");
    }
}

// генерация текста
async function generateLyrics() {
    let title = document.getElementById("songTitle")?.value;
    let outputDiv = document.getElementById("output");

    if (!title || title.trim() === "") {
        outputDiv.innerHTML = "<p style='color: red;'>Введите название трека!</p>";
        outputDiv.classList.add("show");
        return;
    }

    outputDiv.innerHTML = ""; // очищаем вывод
    outputDiv.classList.add("show");

    // Исправлено: добавлены обратные кавычки
    const eventSource = new EventSource(`/generate/stream?prompt=${encodeURIComponent(title)}`);

    eventSource.onmessage = async function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.lyrics) {
                // финальный текст трека
                lastGeneratedLyrics = data.lyrics;

                let formattedLyrics = formatLyrics(data.lyrics);
                outputDiv.innerHTML += `<div class="lyrics">${formattedLyrics}</div>`;

                let likeBtn = document.getElementById("likeBtn");

                if (!likeBtn) {
                    likeBtn = document.createElement("button");
                    likeBtn.id = "likeBtn";
                    likeBtn.innerHTML = "❤️ Сохранить в избранное";
                    likeBtn.style.background = "limegreen";
                    likeBtn.style.width = "100%";
                    likeBtn.style.padding = "12px";
                    likeBtn.style.marginTop = "15px";
                    likeBtn.style.border = "none";
                    likeBtn.style.borderRadius = "5px";
                    likeBtn.style.color = "white";
                    likeBtn.style.fontSize = "16px";
                    likeBtn.style.cursor = "pointer";
                    likeBtn.addEventListener("click", likeLyrics);
                    outputDiv.parentNode.insertBefore(likeBtn, outputDiv.nextSibling);
                }

                likeBtn.style.display = "block";
                eventSource.close(); // завершаем поток
            }
        } catch {
            const step = document.createElement("p");
            step.textContent = event.data;
            outputDiv.appendChild(step);
        }
    };

    eventSource.onerror = function(err) {
        console.error("❌ Ошибка SSE:", err);
        eventSource.close();
        outputDiv.innerHTML += "<p style='color: red;'>Проблема с соединением SSE.</p>";
    };
}

// Исправлено форматирование
function formatLyrics(text) {
    let lines = text.split("\n");
    let formatted = "";

    for (let i = 0; i < lines.length; i++) {
        formatted += `<p>${lines[i]}</p>`;
        if ((i + 1) % 4 === 0) {
            formatted += "<br>";
        }
    }

    return formatted;
}

document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const logoutBtn = document.getElementById("logoutBtn");
    const status = document.getElementById("status");
    const likeBtn = document.getElementById("likeBtn");

    if (likeBtn) {
        likeBtn.addEventListener("click", likeLyrics);
        likeBtn.style.display = "none";
    }

    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            let user = document.getElementById("loginUser").value;
            let pass = document.getElementById("loginPass").value;
    
            try {
                let response = await fetch("/login", {
                    method: "POST",
                    body: new URLSearchParams({ username: user, password: pass })
                });
    
                // Пытаемся получить JSON в любом случае, даже при ошибке
                let data = await response.json().catch(() => {
                    return { message: "Ошибка формата ответа" };
                });
    
                if (response.ok) {
                    status.innerText = data.message;
                    window.location.href = "/generator";
                } else {
                    // Показываем сообщение об ошибке
                    status.innerText = data.message || `Ошибка ${response.status}: Неверное имя пользователя или пароль`;
                    status.style.color = "red";
                }
            } catch (error) {
                console.error("Ошибка при авторизации:", error);
                status.innerText = "Ошибка соединения с сервером";
                status.style.color = "red";
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            let user = document.getElementById("regUser").value;
            let pass = document.getElementById("regPass").value;

            let response = await fetch("/register", {
                method: "POST",
                body: new URLSearchParams({ username: user, password: pass })
            });

            let data = await response.json();
            status.innerText = data.message;
        });
    }

    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            await fetch("/logout", { method: "POST" });
            status.innerText = "👋 Вы вышли!";
            logoutBtn.style.display = "none";
            setTimeout(() => {
                window.location.href = "/";
            }, 1000);
        });
    }
});