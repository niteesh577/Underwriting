// "use client"
// import { supabase } from '@/services/supabaseClient'
// import React, {useEffect, useState} from 'react'

// function Provider({ children }) {

//     useEffect(() => { CreateNewUser(); }, [])
//     const CreateNewUser = () => {

//         supabase.auth.getUser().then(async({data:{user}})=> {

//             // check if user exists
//             let { data: users, error } = await supabase
//            .from('users')
//            .select("*")
//            .eq('email', user?.email);

//            console.log(users)

//             // if not create a new user

//             if(users?.length==0){
//                const {data, error} =  await supabase.from('users').insert([
//                     {
//                         name: user?.user_metadata?.name,
//                         email: user?.email,
//                         picture: user?.user_metadata?.picture

//                     }
//                 ])
//                 console.log(data);
//             }

//         })
        
        
//     }

      
//   return (
//     <div>
//       {children}
//     </div>
//   )
// }

// export default Provider





// "use client"
// import { supabase } from '@/services/supabaseClient'
// import React, { useEffect } from 'react'

// function Provider({ children }) {
  
//   useEffect(() => {
//     // Subscribe to login events
//     const { data: subscription } = supabase.auth.onAuthStateChange(
//       async (event, session) => {
//         console.log("Auth event:", event, "Session:", session);

//         if (event === "SIGNED_IN" && session?.user) {
//           await createNewUser(session.user);
//         }
//       }
//     );

//     // Cleanup listener
//     return () => {
//       subscription.subscription.unsubscribe();
//     };
//   }, []);

//   const createNewUser = async (user) => {
//     console.log("Inside createNewUser:", user);

//     // Check if user exists
//     const { data: Users, error: fetchError } = await supabase
//       .from("Users")
//       .select("*")
//       .eq("email", user.email);

//     if (fetchError) {
//       console.error("Error fetching users:", fetchError.message);
//       return;
//     }

//     console.log("Users fetched:", Users);

//     if (!Users || Users.length === 0) {
//       const { data, error: insertError } = await supabase.from("Users").insert([
//         {
//           id: user.id,
//           name: user.user_metadata?.name,
//           email: user.email,
//           picture: user.user_metadata?.picture,
//         },
//       ]);

//       if (insertError) {
//         console.error("Error inserting user:", insertError.message);
//       } else {
//         console.log("User inserted successfully:", data);
//       }
//     } else {
//       console.log("User already exists in DB");
//     }
//   };

//   return <div>{children}</div>;
// }

// export default Provider;


"use client";
import { useEffect } from "react";
import { supabase } from "@/services/supabaseClient";

export default function Provider({ children }) {
  useEffect(() => {
    // Get initial session and user
    supabase.auth.getSession().then(({ data }) => {
      if (data.session?.user) {
        console.log("Initial session user:", data.session.user);
        upsertUser(data.session.user);
      }
    });

    // Listen for auth state changes
    const { data: listener } = supabase.auth.onAuthStateChange((event, session) => {
      console.log("Auth state event:", event, session);
      if (event === "SIGNED_IN" && session?.user) {
        upsertUser(session.user);
      }
    });

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  const upsertUser = async (user) => {
    console.log("Upsert user:", user);
    const { data: existing, error: selectError } = await supabase
      .from("Users")
      .select("*")
      .eq("email", user.email);

    if (selectError) {
      console.error("Select error:", selectError);
      return;
    }

    if (!existing || existing.length === 0) {
      const { data: inserted, error: insertError } = await supabase.from("Users").insert([
        {
          id: user.id, // ensure table uses id linked to auth
          name: user.user_metadata?.name || user.user_metadata?.full_name,
          email: user.email,
          picture: user.user_metadata?.picture || user.user_metadata?.avatar_url,
        },
      ]);
      if (insertError) console.error("Insert error:", insertError);
      else console.log("User inserted:", inserted);
    } else {
      console.log("User already exists");
    }
  };

  return <>{children}</>;
}
