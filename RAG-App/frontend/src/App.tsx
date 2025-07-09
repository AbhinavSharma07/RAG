import React from 'react';
import Chat from './Chat';
import Upload from './Upload';

export default function App() {
  return (
    <div className="container">
      <h1>📚 RAG Chat</h1>
      <Upload />
      <Chat />
    </div>
  );
}
