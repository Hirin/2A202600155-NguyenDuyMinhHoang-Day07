// Wrapper for API calls
const API = {
    async query(question, filters = {}) {
        // Construct query string with filter info if needed
        // For our API, the filter is embedded in the query if we build a complex query,
        // or we just rely on the query parser logic.
        // Wait, our API endpoints just take: `POST /api/query` -> `{"query": "..."}`
        // The RAG agent extracts metadata filters internally using the QueryParser.
        let queryStr = question;
        if (filters.linh_vuc) queryStr += ` lĩnh vực ${filters.linh_vuc}`;
        if (filters.co_quan) queryStr += ` cơ quan ${filters.co_quan}`;
        if (filters.doi_tuong) queryStr += ` cho ${filters.doi_tuong}`;

        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: queryStr })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Có lỗi xảy ra khi gọi API");
        }
        
        return res.json();
    },
    
    async getProcedure(id) {
        const res = await fetch(`/api/procedure/${id}`);
        if (!res.ok) throw new Error("Không tìm thấy thủ tục");
        return res.json();
    },

    async listProcedures(page = 1) {
        const res = await fetch(`/api/procedures?page=${page}`);
        return res.json();
    },
    
    // Admin APIs
    async getHealth() {
        const res = await fetch("/api/health");
        return res.json();
    }
};

window.API = API;
