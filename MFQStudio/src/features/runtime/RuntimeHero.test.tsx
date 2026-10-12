import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus } from '../../shared/api/contracts/runtime';
import { useJobStore } from '../../stores/jobStore';
import { RuntimeHero } from './RuntimeHero';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, instances: [], studio: null, chinese: false }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string, en: string) => state.chinese ? zh : en }) }));

function Location() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.hash}</output>;
}

beforeEach(() => { state.runtime = null; state.chinese = false; useJobStore.getState().setJobs([]); });

it.each([false, true])('opens local checkpoints from the empty-model Load action (Chinese: %s)', chinese => {
  state.chinese = chinese;
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.queryByRole('button', { name: chinese ? '模型' : 'Models' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: chinese ? '加载' : 'Load' }));
  expect(screen.getByTestId('location')).toHaveTextContent('/models#local-checkpoints');
});

it('retains the normal Models action when a model is running', () => {
  state.runtime = { model: 'model-a', runtime_state: 'ready' };
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.queryByRole('button', { name: 'Load' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Models' }));
  expect(screen.getByTestId('location').textContent).toBe('/models');
});
