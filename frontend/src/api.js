export async function uploadFiles(files, overrides = {}) {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }
    formData.append("overrides", JSON.stringify(overrides));
  
    const res = await fetch("http://127.0.0.1:8000/underwrite", {
      method: "POST",
      body: formData,
    });
  
    if (!res.ok) throw new Error("Failed to fetch analysis");
    return res.json();
  }
  