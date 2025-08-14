async function getStatus() {
    let res = await fetch('/api/status');
    let data = await res.json();
    document.getElementById('currentFile').innerText = data.file;
    document.getElementById('progress').innerText = data.percent.toFixed(1) + '%';
    document.getElementById('temperature').innerText = data.temperature.toFixed(1) + '°C';
}

async function listFiles() {
    let res = await fetch('/api/files');
    let data = await res.json();
    let table = document.getElementById('fileTable');
    table.innerHTML = '';
    data.files.forEach(f => {
        let row = table.insertRow();
        row.insertCell(0).innerText = f.name;
        let delBtn = document.createElement('button');
        delBtn.innerText = '删除';
        delBtn.onclick = () => deleteFile(f.name);
        row.insertCell(1).appendChild(delBtn);
        let playBtn = document.createElement('button');
        playBtn.innerText = '播放';
        playBtn.onclick = () => selectFile(f.name);
        row.insertCell(2).appendChild(playBtn);
    });
}

async function uploadFile() {
    let fileInput = document.getElementById('uploadFile');
    let formData = new FormData();
    formData.append('video', fileInput.files[0]);
    await fetch('/api/upload', { method: 'POST', body: formData });
    listFiles();
}

async function deleteFile(name) {
    await fetch('/api/files/' + encodeURIComponent(name), { method: 'DELETE' });
    listFiles();
}

async function selectFile(name) {
    await fetch('/api/control/select', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({file: name})});
}

async function sendControl(cmd) {
    await fetch('/api/control/' + cmd, { method: 'POST' });
}

setInterval(getStatus, 2000);
listFiles();
