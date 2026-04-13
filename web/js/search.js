document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const filterForm = document.getElementById('filter-form');
    
    const loadingEl = document.getElementById('loading');
    const answerContainer = document.getElementById('answer-container');
    const answerText = document.getElementById('answer-text');
    const factsTable = document.getElementById('facts-table').querySelector('tbody');
    const statusBadge = document.getElementById('status-badge');
    const citationsContainer = document.getElementById('citations-container');
    const sourcePreview = document.getElementById('source-preview');

    // Mappings for facts translation
    const FACTS_MAP = {
        'ma_thu_tuc': 'Mã thủ tục',
        'ten_thu_tuc': 'Tên thủ tục',
        'thoi_han': 'Thời hạn',
        'phi_le_phi': 'Phí, lệ phí',
        'ho_so': 'Hồ sơ',
        'co_quan': 'Cơ quan thực hiện',
        'can_cu': 'Căn cứ pháp lý'
    };

    // Check URL params on load
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    if (q) {
        queryInput.value = q;
        submitQuery(q);
    }

    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = queryInput.value.trim();
        if (!q) return;
        
        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('q', q);
        window.history.pushState({}, '', url);
        
        submitQuery(q);
    });

    async function submitQuery(query) {
        // Collect filters
        const formData = new FormData(filterForm);
        const filters = Object.fromEntries(formData.entries());

        loadingEl.classList.remove('hidden');
        answerContainer.classList.add('hidden');
        sourcePreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Đang lấy dữ liệu nguồn...</p>';

        try {
            const result = await API.query(query, filters);
            renderAnswer(result);
        } catch (err) {
            console.error(err);
            renderError(err.message);
        } finally {
            loadingEl.classList.add('hidden');
        }
    }

    function renderAnswer(data) {
        // Render Text
        answerText.innerHTML = data.answer.replace(/\n/g, '<br>');
        
        // Render Status Badge
        statusBadge.className = ''; // reset
        if (data.status === 'grounded') {
            statusBadge.className = 'badge badge-success';
            statusBadge.textContent = '✓ An toàn & Đủ dữ kiện';
        } else if (data.status === 'conflict') {
            statusBadge.className = 'badge badge-danger';
            statusBadge.textContent = '⚠ Có xung đột nguồn';
        } else {
            statusBadge.className = 'badge badge-warning';
            statusBadge.textContent = '! Thiếu thông tin';
        }
        
        // Render Facts
        factsTable.innerHTML = '';
        if (data.facts && Object.keys(data.facts).length > 0) {
            for (const [key, value] of Object.entries(data.facts)) {
                if (value && ['ma_thu_tuc', 'ten_thu_tuc', 'thoi_han', 'phi_le_phi', 'co_quan'].includes(key)) {
                    // Make Ma Thu Tuc a link if it exists
                    let valHtml = value;
                    if (key === 'ma_thu_tuc' && data.facts.source_url) {
                       valHtml = `<a href="${data.facts.source_url}" target="_blank" title="Xem trên Cổng DVC QG">${value} ↗</a>`;
                    } else if (key === 'ma_thu_tuc') {
                       // Fallback internal link
                       valHtml = `<a href="/procedure.html?id=${value}" target="_blank">${value}</a>`;
                    }

                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <th>${FACTS_MAP[key] || key}</th>
                        <td>${valHtml}</td>
                    `;
                    factsTable.appendChild(tr);
                }
            }
        }
        
        // Render Citations
        citationsContainer.innerHTML = '';
        if (data.citations && data.citations.length > 0) {
            data.citations.forEach(cit => {
                const button = document.createElement('button');
                button.className = 'chip';
                button.textContent = cit.replace('[', '').replace(']', '');
                button.onclick = () => loadSourceForCitation(cit);
                citationsContainer.appendChild(button);
            });
            // Auto click first citation
            loadSourceForCitation(data.citations[0]);
        } else {
            citationsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.9rem;">Không có trích dẫn cụ thể.</span>';
            sourcePreview.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Không tìm thấy nguồn liên quan để hiển thị.</p>';
        }

        answerContainer.classList.remove('hidden');
    }

    function renderError(msg) {
        answerText.innerHTML = `<span style="color: var(--danger-color);">${msg}</span>`;
        statusBadge.className = 'badge badge-danger';
        statusBadge.textContent = '⚠ Lỗi hệ thống';
        factsTable.innerHTML = '';
        citationsContainer.innerHTML = '';
        sourcePreview.innerHTML = '';
        answerContainer.classList.remove('hidden');
    }

    async function loadSourceForCitation(citationToken) {
        // Token looks like: [1.00309|thoi_han_giai_quyet]
        const raw = citationToken.replace('[', '').replace(']', '').trim();
        const parts = raw.split('|');
        if (parts.length !== 2) return;
        
        const maThuTuc = parts[0];
        const sectionType = parts[1];

        sourcePreview.innerHTML = `Đang tải tài liệu ${maThuTuc}... <div class="loader" style="width:20px;height:20px;border-width:2px;margin:10px 0;"></div>`;

        try {
            const doc = await API.getProcedure(maThuTuc);
            const section = doc.sections.find(s => s.section_type === sectionType || s.heading.toLowerCase().includes(sectionType.replace(/_/g, ' ')));
            
            if (section) {
                // Formatting for display
                const headingHtml = `<h4 style="margin-top:0; color:var(--primary-color); border-bottom:1px solid var(--border-color); padding-bottom:8px;">${section.heading}</h4>`;
                
                // Add some basic markdown parsing (newlines to br, lists, etc)
                let contentHtml = section.content.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
                contentHtml = `<p>${contentHtml}</p>`;

                // If internal URL exists add button
                let actionsHtml = '';
                actionsHtml = `<div style="margin-top: 20px; padding-top: 15px; border-top: 1px dashed #ccc;">
                    <a href="/procedure.html?id=${doc.ma_thu_tuc}" class="btn btn-primary btn-sm">Xem toàn văn thủ tục nội bộ</a>`;
                
                if (doc.source_url) {
                    actionsHtml += ` <a href="${doc.source_url}" target="_blank" class="btn btn-primary btn-sm" style="background:#0f172a;margin-left:8px;">Cổng DVC QG ↗</a>`;
                }
                actionsHtml += `</div>`;


                sourcePreview.innerHTML = headingHtml + contentHtml + actionsHtml;
            } else {
                sourcePreview.innerHTML = `<h4>Lỗi</h4><p>Không tìm thấy mục <strong>${sectionType}</strong> trong tài liệu <strong>${maThuTuc}</strong>.</p>`;
            }
        } catch (e) {
            sourcePreview.innerHTML = `<p style="color: var(--danger-color);">Lỗi tải nguồn: ${e.message}</p>`;
        }
    }
});
