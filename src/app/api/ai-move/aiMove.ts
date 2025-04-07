import { spawn } from 'child_process';

export async function runAiMove(state: string): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const pyProcess = spawn('python3', ['./othello-rl/a3c_agent.py', state], { cwd: process.cwd() });
    let output = '';

    pyProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      console.error('Python error:', data.toString());
    });

    pyProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}`));
      } else {
        resolve(output.trim());
      }
    });
  });
}
