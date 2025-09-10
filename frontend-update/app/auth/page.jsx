// "use client"
// import { Button } from '@/components/ui/button'
// import { supabase } from '@/services/supabaseClient'
// import Image from 'next/image'
// import React from 'react'

// function Login() {

//   const signInWithGoogle =async () => {
//     const {error} = await supabase.auth.signInWithOAuth({
//       provider: 'google',

//     })

//     if (error) {
//       console.error('Error:', error.message)
//     }

//   }
//   return (
//     <div className='flex flex-col items-center justify-center h-screen'>
//       <div className='flex flex-col items-center border rounded-2xl p-8'>
//         <Image src={"/logo.png"} alt="logo" width={400} height={100} className='w-[120px]' />
     
//       <div>
//         <Image src = {'/login.png'} alt = 'login' width = {500} height = {500} className = 'w-[400px] h-[250px] border rounded-2xl'/>
//         <h2 className='text-2xl font-bold text-center mt-5'>Welcome to Resume Parser</h2>
//         <p className='text-gray-500 text-center'>Sign in to get started</p>
//         <Button className='mt-7 w-full' onClick={signInWithGoogle}> Login with Google</Button>
//       </div>
//     </div>
//     </div>
//   )
// }

// export default Login








// "use client";
// import { useState } from "react";
// import { supabase } from "@/services/supabaseClient";
// import { Button } from "@/components/ui/button";
// import Image from "next/image";
// import { useRouter } from "next/navigation";

// export default function AuthPage() {
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const router = useRouter();

//   const signInWithGoogle = async () => {
//     const { error } = await supabase.auth.signInWithOAuth({ provider: "google" });
//     if (error) console.error("OAuth error:", error.message);
//   };

//   const signUpWithEmail = async () => {
//     const { data, error } = await supabase.auth.signUp({
//       email,
//       password,
//       options: { emailRedirectTo: window.location.origin },
//     });

//     if (error) {
//       console.error("Email sign-up error:", error.message);
//     } else {
//       console.log("Sign-up success:", data);

//       // ⚡ Redirect to /home after success
//       router.push("/home");
//     }
//   };

//   return (
//     <div className="flex flex-col items-center justify-center h-screen p-4">
//       <div className="text-center mb-8">
//         <Image src="/logo.png" alt="logo" width={120} height={120} />
//         <h2 className="text-2xl font-bold mt-4">Welcome to Acquire Underwriting</h2>
//         <p className="text-gray-600 mt-2">Sign in or sign up to continue</p>
//       </div>
//       <div className="space-y-4 w-full max-w-sm">
//         <input
//           type="email"
//           placeholder="Email"
//           value={email}
//           onChange={(e) => setEmail(e.target.value)}
//           className="w-full rounded-md border px-4 py-2"
//         />
//         <input
//           type="password"
//           placeholder="Password"
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           className="w-full rounded-md border px-4 py-2"
//         />
//         <Button onClick={signUpWithEmail} className="w-full">
//           Sign Up with Email
//         </Button>
//         <div className="text-center text-gray-500">or</div>
//         <Button variant="outline" onClick={signInWithGoogle} className="w-full">
//           Continue with Google
//         </Button>
//       </div>
//     </div>
//   );
// }

"use client";
import { useState } from "react";
import { supabase } from "@/services/supabaseClient";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import { useRouter } from "next/navigation";

export default function AuthPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const signInWithGoogle = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/home`,
      },
    });

    if (error) {
      console.error("OAuth error:", error.message);
    }
  };

  const signUpWithEmail = async () => {
    if (!name) {
      alert("Please enter your name");
      return;
    }

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { name },
        emailRedirectTo: `${window.location.origin}/home`,
      },
    });

    if (error) {
      console.error("Email sign-up error:", error.message);
    } else if (data?.user) {
      console.log("Sign-up success:", data.user);

      await sendUserToBackend({
        ...data.user,
        user_metadata: { ...data.user.user_metadata, name },
      });

      router.push("/home");
    }
  };

  const sendUserToBackend = async (user) => {
    try {
      const response = await fetch(
        "https://underwriting-at5l.onrender.com/save_user",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: user.email,
            name: user.user_metadata?.name || "",
            picture: user.user_metadata?.picture || "",
          }),
        }
      );

      if (!response.ok) {
        console.error("Failed to send user to backend");
      } else {
        console.log("User successfully sent to backend");
      }
    } catch (err) {
      console.error("Error sending user to backend:", err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen p-4">
      <div className="text-center mb-8">
        <div className="flex justify-center">
          <Image src="/login.png" alt="logo" width={220} height={120} />
        </div>
        <h2 className="text-2xl font-bold mt-4">
          Welcome to Acquire Underwriting
        </h2>
        <p className="text-gray-600 mt-2">Sign in or sign up to continue</p>
      </div>
      <div className="space-y-4 w-full max-w-sm">
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full rounded-md border px-4 py-2"
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-md border px-4 py-2"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-md border px-4 py-2"
        />
        <Button onClick={signUpWithEmail} className="w-full">
          Sign Up with Email
        </Button>
        <div className="text-center text-gray-500">or</div>
        <Button variant="outline" onClick={signInWithGoogle} className="w-full">
          Continue with Google
        </Button>
      </div>
    </div>
  );
}
