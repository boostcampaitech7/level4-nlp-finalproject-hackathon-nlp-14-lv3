import { DeepChat } from "deep-chat-react";

export const DeepChatComponent = () => {

  return (
    <div>
      <h1>Deep Chat Integration</h1>
      <DeepChat
        style={{ borderRadius: "10px", border: "1px solid #ccc", padding: "10px" }}
        textInput={{ placeholder: { text: "Type your message here..." } }}

		connect={{
			"url": "https://localhost:8000/inference",
			"method": "POST",
		  }}
      />
    </div>
  );
};


// import { useState } from "react";
// import { DeepChat } from "deep-chat-react";

// export const DeepChatComponent = () => {
//   const [history] = useState([
//     {
//       text: "Hello! How can I assist you today?",
//       role: "ai",
//     },
//   ]);

//   return (
//     <div>
//       <h1>Deep Chat</h1>
//       <DeepChat
//         style={{
//           borderRadius: "10px",
//           border: "1px solid #ccc",
//           padding: "10px",
//         }}
//         connect={{
//           url: "http://localhost:5173/items/chat", // 백엔드 API URL
//           method: "POST",
//           headers: {
//             "Content-Type": "application/json",
//             // Authorization: "Bearer your_api_token", // 필요한 경우 인증 토큰 추가
//           },
//         //   additionalBodyProps: {
//         //     customField: "customValue",
//         //   },
//         //   credentials: "same-origin",
//         }}
//         textInput={{
//           placeholder: { text: "Type your message here..." },
//         }}
//         history={history}
//       />
//     </div>
//   );
// };

// export default DeepChatComponent;
