import * as fs from 'fs';

const input_01 : string = 'input/day01_01.txt';
const fileContent: string = fs.readFileSync(input_01, 'utf-8');

// First start
// - The dial starts by pointing at 50.
// - A rotation starts with an L or R which indicates whether the rotation should be to the left (toward lower numbers) or to the right (toward higher numbers).
// - Because the dial is a circle, turning the dial left from 0 one click makes it point at 99. Similarly, turning the dial right from 99 one click makes it point at 0.

function countDialIsZeroAfterRotations(lines : string): number {
    let dialCounting: number = 0;
    let currentState: number = 50;

    lines.split('\n').forEach(line => {
        const direction: string = line[0];
        const steps: number = parseInt(line.slice(1));

        if (direction === 'R') {
            currentState = (currentState + steps) % 100;
        } else if (direction === 'L') {
            currentState = (currentState - steps + 100) % 100;
        }

        if (currentState === 0) {
            dialCounting++;
        }
    });
    
    return dialCounting
}



console.log(countDialIsZeroAfterRotations(fileContent));