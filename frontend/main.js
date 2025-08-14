// async function getStatus() {
//     let res = await fetch('/api/status');
//     let data = await res.json();

//     // 显示文件名
//     document.getElementById('currentFile').innerText = data.file;

//     // 百分比 + 帧数显示
//     document.getElementById('progress').innerText = 
//         `${data.percent.toFixed(1)}% (${data.current_frame}/${data.total_frames})`;

//     document.getElementById('temperature').innerText = data.temperature.toFixed(1) + '°C';

//     // 更新进度条
//     let seekBar = document.getElementById('seekBar');
//     if (seekBar && data.total_frames > 0) {
//         seekBar.max = data.total_frames;
//         seekBar.value = data.current_frame;
//     }
// }

