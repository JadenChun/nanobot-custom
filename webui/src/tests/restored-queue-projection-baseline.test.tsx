import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ThreadComposer } from '@/components/thread/ThreadComposer';
afterEach(()=>{cleanup();localStorage.clear();});
describe('restored queued input must not become a fresh submission',()=>{
 it.each(['queue-key switch','component remount'])('keeps an unresolved attempt held after %s',(transition)=>{
  const onSend=vi.fn(()=>({turnId:'attempt-original',userMessageId:'user-original',sideChannel:false,admission:new Promise<boolean>(()=>{})}));
  const view=(key:string)=><ThreadComposer onSend={onSend} isStreaming pendingQueueKey={key}/>;
  const mounted=render(view('chat-a'));
  fireEvent.change(screen.getByLabelText('Message input'),{target:{value:'do this exactly once'}});
  fireEvent.keyDown(screen.getByLabelText('Message input'),{key:'Enter'});
  fireEvent.click(screen.getByRole('button',{name:'Steer',exact:true}));
  expect(onSend).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button',{name:'Steer',exact:true})).toBeDisabled();
  if(transition==='queue-key switch'){mounted.rerender(view('chat-b'));mounted.rerender(view('chat-a'));}
  else {mounted.unmount();render(view('chat-a'));}
  fireEvent.click(screen.getByRole('button',{name:'Steer',exact:true}));
  expect(onSend).toHaveBeenCalledTimes(1);
 });
});
