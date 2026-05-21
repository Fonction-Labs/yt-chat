// Background script
chrome.runtime.onInstalled.addListener(function() {
  console.log('Extension installed.');
});

chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
  $.get("http://localhost:8000/copilot/index.js", function(scriptContent) {
    const codeToExecute = `
      ${scriptContent}
      window.mountChainlitWidget({
        chainlitServer: "http://localhost:8000"
      });
    `;
    chrome.tabs.executeScript(tabs[0].id, { code: codeToExecute });
  });
});
