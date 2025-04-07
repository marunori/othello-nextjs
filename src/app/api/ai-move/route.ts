import { NextResponse } from 'next/server';
import { runAiMove } from './aiMove';

export async function GET(request: Request): Promise<Response> {
  // Extract optional game state from query parameters.
  const { searchParams } = new URL(request.url);
  const state = searchParams.get('state') || '';

  try {
    const move = await runAiMove(state);
    return NextResponse.json({ move });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unknown error' }, { status: 500 });
  }
}
