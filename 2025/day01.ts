import * as fs from 'fs';

const input_01 : string = 'input/day01_01.txt';

try {
  // Synchronous reading
  const fileContent: string = fs.readFileSync(input_01, 'utf-8');
  console.log('Synchronous file content:', fileContent);

  // Asynchronous reading
  fs.readFile(input_01, 'utf-8', (err, data) => {
    if (err) {
      console.error('Error reading file asynchronously:', err);
      return;
    }
    console.log('Asynchronous file content:', data);
  });
} catch (error) {
  console.error('Error reading file synchronously:', error);
}