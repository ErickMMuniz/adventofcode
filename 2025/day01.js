"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var input_01 = 'input/day01_01.txt';
try {
    // Synchronous reading
    var fileContent = fs.readFileSync(input_01, 'utf-8');
    console.log('Synchronous file content:', fileContent);
    // Asynchronous reading
    fs.readFile(input_01, 'utf-8', function (err, data) {
        if (err) {
            console.error('Error reading file asynchronously:', err);
            return;
        }
        console.log('Asynchronous file content:', data);
    });
}
catch (error) {
    console.error('Error reading file synchronously:', error);
}
