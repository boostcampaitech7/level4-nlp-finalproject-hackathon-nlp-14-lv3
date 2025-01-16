import { DeepChat } from "deep-chat-react";

export const DeepChatComponent = () => {
  const history = [
    { role: "user", text: "Hey, how are you today?" },
    { role: "ai", text: "I am doing very well!" },
  ];

  return (
    <div>
      <h1>Deep Chat Integration</h1>
      <DeepChat
        demo={true}
        style={{ borderRadius: "10px", border: "1px solid #ccc", padding: "10px" }}
        textInput={{ placeholder: { text: "Type your message here..." } }}
        history={history}
      />
    </div>
  );
};
