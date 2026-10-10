import React from 'react';

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`max-w-[85%] md:max-w-[75%] px-5 py-3.5 rounded-2xl whitespace-pre-wrap text-sm leading-relaxed shadow-sm ${
          isUser 
            ? 'bg-blue-600 text-white rounded-br-sm' 
            : 'bg-white text-slate-800 rounded-bl-sm border border-slate-200'
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}