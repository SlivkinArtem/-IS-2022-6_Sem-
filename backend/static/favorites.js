
document.addEventListener('DOMContentLoaded', function() {
    // Загрузка избранных текстов
    fetchLikedTexts();
    
    // Обработчик для кнопки выхода
    document.getElementById('logoutBtn').addEventListener('click', function() {
        fetch('/logout', {
            method: 'POST',
            credentials: 'include'
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            window.location.href = '/';
        })
        .catch(error => {
            console.error('Ошибка при выходе:', error);
        });
    });
});

// Функция для форматирования текста песни (разбивка на строки)
function formatLyrics(text) {
    // Заменяем переносы строк на HTML-теги <br>
    return text.replace(/\n/g, '<br>');
}

// Функция для форматирования даты
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Функция для получения избранных текстов
function fetchLikedTexts() {
    fetch('/liked', {
        method: 'GET',
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 401) {
                window.location.href = '/'; // Перенаправление на страницу входа
                throw new Error('Необходимо авторизоваться');
            }
            throw new Error('Ошибка при получении данных');
        }
        return response.json();
    })
    .then(data => {
        displayLikedTexts(data.liked_texts);
    })
    .catch(error => {
        console.error('Ошибка:', error);
        document.getElementById('likedTextsContainer').innerHTML = 
            `<p class="error">Ошибка при загрузке: ${error.message}</p>`;
    });
}

// Функция для отображения избранных текстов
function displayLikedTexts(texts) {
    const container = document.getElementById('likedTextsContainer');
    
    if (!texts || texts.length === 0) {
        container.innerHTML = `
            <div class="empty-message">
                <p>У вас пока нет сохраненных текстов</p>
                <button onclick="window.location.href='/generator'">Перейти к генератору</button>
            </div>
        `;
        return;
    }
    
    // Сортировка по дате (новые сверху)
    texts.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    let html = '';
    
    texts.forEach(item => {
        html += `
            <div class="liked-item" data-id="${item.id}">
                <div class="lyrics">${formatLyrics(item.text_content)}</div>
                <div class="liked-date">Сохранено: ${formatDate(item.created_at)}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}